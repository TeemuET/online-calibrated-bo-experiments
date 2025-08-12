import json
import string
from imports.general import *
from imports.ml import *
from dataclasses import dataclass, asdict
import warnings


@dataclass
class Parameters:
    surrogate: str = "GP"  # surrogate function name
    acquisition: str = "EI"  # acquisition function name
    recal_mode: str = "cv"
    data_name: str = "Benchmark"  # dataclass name
    seed: bool = 0  # random seed
    d: int = 1  # number of input dimensions
    n_test: int = 5000  # number of test samples for calibration analysis
    n_initial: int = 5  # number of starting points
    n_validation: int = 100  # number of iid samples for recalibration
    n_evals: int = 5  # number of BO iterations
    n_pool : int = 5000
    rf_cv_splits: int = 2  # number of CV splits for random forest hyperparamtuning
    vanilla: bool = False  # simplest implementation (used for test)
    plot_it: bool = False  # whether to plot during BO loop
    save_it: bool = True  # whether to save progress
    bo: bool = False  # performing bo to sample X or merely randomly sample X
    noisify: bool = True
    fix_surrogate_logic = True # bug fix for the logic of the surrogate model in experiments.yp
    track_surrogate_state: bool = False  # whether to track surrogate state
    test: bool = True
    beta: float = 1.0 #beta value if acquisition function is UCB. Experimenting with different values seem to indicate that beta = 1 is best, but this is probably largely dependant on optim. problem. 
    recalibrate: bool = False
    analyze_all_epochs: bool = True
    extensive_metrics: bool = True
    maximization: bool = False
    fully_bayes: bool = False  # if fully bayes in BO rutine (marginalize hyperparams)
    xi: float = 0.0  # exploration parameter for BO
    device: str = "cpu"  # Add this line to control device (cpu or cuda)
    problem: str = ""  # e.g. "Alpine01" # subproblem name, overwrites problem_idx
    problem_idx: int = 0
    prob_acq: bool = False  # if acqusition function should sample like a prob dist. If False, argmax is used.
    std_change: float = 1.0  # how to manipulate predictive std
    snr: float = 1000.0
    sigma_data: float = None  # follows from problem
    sigma_noise: float = None  # computed as function of SNR and sigma_data
    n_calibration_bins: int = 20
    K: int = 1  # number of terms in sum for VerificationData
    b_train: int = 64 # Batch size while training NN on MNIST
    hidden_size: int = 100 # hidden layer number of neurons for NN on MNIST
    savepth: str = os.getcwd() + "/results/"
    experiment: str = ""  # folder name
    n_seeds_per_job: int = 1 #Select how many jobs to run for this particular seed. Set via input params only.
    save_scratch: bool = False #If want results saved on scratch directory.

    def __init__(self, kwargs: Dict = {}, mkdir: bool = False) -> None:
        self.update(kwargs)
        if self.surrogate == "RS" and (self.recalibrate or self.bo):
            sys.exit(0)
        self.acquisition = "RS" if self.surrogate == "RS" else self.acquisition

        if self.problem == "" and self.data_name.lower() == "benchmark":
            problem = self.find_benchmark_problem_i()
            kwargs["problem"] = problem
            kwargs['savepth'] = "./results_synth_data/"

        elif self.data_name.lower() == "mnist":
            kwargs["problem"] = "mnist"
            kwargs["d"] = 5
            kwargs['savepth'] = "./results_real_data/results_mnist/"
        elif self.data_name.lower() == "fashionmnist":
            kwargs["problem"] = "fashionmnist"
            kwargs["d"] = 5
            kwargs['savepth'] = "./results_real_data/results_FashionMNIST/"
        elif self.data_name.lower() == "fashionmnist_cnn":
            kwargs["problem"] = "fashionmnist_cnn"
            kwargs["d"] = 5
            kwargs['savepth'] = "./results_real_data/results_FashionMNIST_CNN/"
        elif self.data_name.lower() == "mnist_cnn":
            kwargs["problem"] = "mnist_cnn"
            kwargs["d"] = 5
            kwargs['savepth'] = "./results_real_data/results_MNIST_CNN/"
        elif self.data_name.lower() == "news":
            kwargs["problem"] = "news"
            kwargs["d"] = 5
            kwargs['savepth'] = "./results_real_data/results_News/"
        elif self.data_name.lower() == "svm_wine":
            kwargs["problem"] = "svm_wine"
            kwargs["d"] = 2
            kwargs['savepth'] = "./results_real_data/results_SVM/"
        if self.save_scratch and self.std_change == 1.0:
            kwargs['savepth'] = kwargs['savepth'].replace(".", "/work3/mikkjo/unibo_results")
        elif self.save_scratch and self.std_change != 1.0:
            kwargs['savepth'] = kwargs['savepth'].replace(".", "/work3/mikkjo/unibo_results/std_change")
        self.update(kwargs)

        # New path creation logic
        base_save_path = self.savepth

        if self.test:
            # For tests, keep it simple in one folder
            folder_name = f"test-{self.experiment}-p{self.problem_idx}-s{self.seed}"
            full_path = os.path.join(base_save_path, folder_name)
        else:
            # For full runs, create a structured, hierarchical path
            # e.g., ./results_synth_data/GP-EI-BENCHMARKS/problem_15/seed_3/
            full_path = os.path.join(
                base_save_path,
                self.experiment,
                f"problem_{self.problem_idx}",
                f"seed_{self.seed}",
            )

        # Set the final save path, ensuring it ends with a separator
        setattr(self, "savepth", os.path.join(full_path, ""))
        
        if mkdir:
            os.makedirs(self.savepth, exist_ok=True)
            self.save()

    def update(self, kwargs, save=False) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Parameter {key} not found")
        if save:
            self.save()

    def find_benchmark_problem_i(self) -> str:
        with open("datasets/benchmarks/unibo-problems-flat.json") as f:
            problems = json.load(f)

        if self.problem_idx >= len(problems):
            raise IndexError(f"problem_idx {self.problem_idx} is out of range for {len(problems)} problems.")

        problem_config = problems[self.problem_idx]
        self.d = problem_config["dim"]  # Set the dimension from the file
        return problem_config["name"]   # Return the problem name

    def save(self) -> None:
        json_dump = json.dumps(asdict(self))
        with open(self.savepth + "parameters.json", "w") as f:
            f.write(json_dump)

    def __post_init__(self):
        if self.device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA device requested but not available. Falling back to CPU.")
            self.device = "cpu"
            