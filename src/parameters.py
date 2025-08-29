import json
import string
from imports.general import *
from imports.ml import *
from dataclasses import dataclass, asdict
import warnings


@dataclass
class Parameters:
    """
    Configuration parameters for the Bayesian Optimization experiment.

    This dataclass holds all the settings for the surrogate model,
    acquisition function, dataset, and the BO loop itself.
    """
    surrogate: str = "GP"                   # Surrogate function (only GP implemented now)
    acquisition: str = "EI"                 # Acqiuisition function name (UCB and EI implemented)
    recal_mode: str = "cv"                  # Calibration data selection mode: "cv" (Leave-one-out cross-validation), "kfold" (K-Fold cross-validation), "iid" (independent calibration set)
    data_name: str = "Benchmark"            # Dataclass name (only synthetic benchmarks functions implemented)
    seed: bool = 0                          # Random seed for experiment reproducibility
    d: int = 1                              # Benchmark input dimension
    n_test: int = 5000                      # Number of test global test points for metric analysis
    n_initial: int = 5                      # Number of starting points in BO
    n_validation: int = 100                 # Number of iid samples for a calibration set if recal_mode is "iid"
    n_evals: int = 50                       # Number of BO iterations
    n_pool : int = 5000                     # Number of pool points, i.e., candidate/domain points for BO
    bo: bool = True                         # Performing BO to sample X or merely randomly sample X
    scale_kernel = True                     # Whether to use a scale kernel in addition to RBF kernel for the GP surrogate
    noisify: bool = True                    # Whether to add noise to the objective function observations
    eta: float = 0.5                        # Learning rate value for recalibration
    recalibrator_type: str = "UNIBOv2"      # Type of recalibrator, e.g., "UNIBOv2", "ONLINEv2", "None"
    recalibration_method: str = "isotonic"  # Recalibration interpolation function: "isotonic" or "gp"
    track_surrogate_state: bool = False     # Whether to print the surrogate model state during the BO loop (used to debug)
    test: bool = True                       # Whether to write results to a test folder or to a full experiment folder
    beta: float = 1.0                       # Beta parameter for UCB acquisition function. Only used if quantile_level is not set. Might be deprecated.
    quantile_level = 0.95                   # Quantile level for UCB and recalibration. That is, if we recalibrate for 0.95, we want the 95% predictive interval to be calibrated and consequenlty the UCB acquisition function uses the 0.95 quantile.
    recalibrate: bool = False               # Whether to perform recalibration or vanilla BO
    analyze_all_epochs: bool = True         # Whether to compute metrics for all epochs or only the final epoch
    extensive_metrics: bool = True          # Whether to compute extensive metrics
    maximization: bool = False              # Whether the objective function is a maximization or minimization problem
    device: str = "cpu"                     # Controlling for the device (cpu or cuda). Code has not been tested on cuda.
    problem: str = ""                       # Problem name, e.g., "Forrester", "SixHumpCamel". Overwrites problem_idx if set.
    problem_idx: int = 0                    # Problem index in the benchmark problems json file. Used only if problem is not set.
    std_change: float = 1.0                 # How to manipulate predictive std
    snr: float = 100.0                      # Signal-to-noise ratio for noisified observations
    n_calibration_bins: int = 20            # Number of calibration bins for the calibration metrics
    savepth: str = os.getcwd()+"/results/"  # Base path to save results
    experiment: str = ""                    # Experiment folder name, e.g., "GP-UCB-10init-recal_None"
    n_seeds_per_job: int = 1                #Select how many jobs to run for this particular seed. Set via input params only.
    save_scratch: bool = False              #If want results saved on scratch directory.

    def __init__(self, kwargs: Dict = {}, mkdir: bool = False) -> None:
        self.update(kwargs)
        if self.surrogate == "RS" and (self.recalibrate or self.bo):
            sys.exit(0)
        self.acquisition = "RS" if self.surrogate == "RS" else self.acquisition

        if self.problem == "" and self.data_name.lower() == "benchmark":
            problem = self.find_benchmark_problem_i()
            kwargs["problem"] = problem
            kwargs['savepth'] = "./results_synth_data/"
        self.update(kwargs)

        base_save_path = self.savepth

        if self.test:
            folder_name = f"test-{self.experiment}-p{self.problem_idx}-s{self.seed}"
            full_path = os.path.join(base_save_path, folder_name)
        else:
            
            problem_folder = f"{self.problem}{self.d}D"
            
            full_path = os.path.join(
                base_save_path,
                self.experiment,       
                problem_folder,          
                f"seed_{self.seed}",
            )
            
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
        self.d = problem_config["dim"]
        return problem_config["name"]

    def save(self) -> None:
        json_dump = json.dumps(asdict(self))
        with open(self.savepth + "parameters.json", "w") as f:
            f.write(json_dump)

    def __post_init__(self):
        if self.device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA device requested but not available. Falling back to CPU.")
            self.device = "cpu"
