from acquisitions.fully_bayes import FullyBayes
from acquisitions.random_search import RandomSearch
from src.dataset import Dataset
from src.parameters import *
from surrogates.deep_ensemble import DeepEnsemble
from surrogates.dummy_surrogate import DummySurrogate
from surrogates.gaussian_process import GaussianProcess
from surrogates.random_forest import RandomForest
from surrogates.bayesian_neural_network import BayesianNeuralNetwork
from botorch.generation.sampling import MaxPosteriorSampling
from botorch.optim import optimize_acqf
from acquisitions.botorch_acqs import (
    ExpectedImprovement,
    UpperConfidenceBound,
)
from src.recalibrator import Recalibrator
from acquisitions.min_posterior_sampling import MinPosteriorSampling
import warnings


class Optimizer(object):
    """# Optimizer class"""

    def __init__(self, parameters: Parameters) -> None:
        self.__dict__.update(asdict(parameters))
        self.parameters = parameters
        self.is_fitted = False

    def fit_surrogate(self, dataset: Dataset) -> None:
        if self.surrogate == "GP":
            self.surrogate_object = GaussianProcess(self.parameters, dataset)
            self.surrogate_model = self.surrogate_object.model
        elif self.surrogate == "RF":
            self.surrogate_object = RandomForest(self.parameters, dataset)
            self.surrogate_model = self.surrogate_object
        elif self.surrogate == "BNN":
            self.surrogate_object = BayesianNeuralNetwork(self.parameters, dataset)
            self.surrogate_model = self.surrogate_object
        elif self.surrogate == "DE":
            self.surrogate_object = DeepEnsemble(self.parameters, dataset)
            self.surrogate_model = self.surrogate_object
        elif self.surrogate == "DS":
            self.surrogate_object = DummySurrogate(self.parameters, dataset)
            self.surrogate_model = self.surrogate_object
        elif self.surrogate == "RS":
            self.surrogate_object = None
            self.surrogate_model = None
        else:
            raise ValueError(f"Surrogate function {self.surrogate} not supported.")
        self.is_fitted = True

    def construct_acquisition_function(
        self, dataset: Dataset, recalibrator: Recalibrator = None
    ) -> None:
        if not self.is_fitted:
            raise RuntimeError("Surrogate has not been fitted!")
        y_opt_tensor = torch.tensor(dataset.y_opt.squeeze())
        if self.acquisition == "EI":
            self.acquisition_function = ExpectedImprovement(
                self.surrogate_model,
                best_f=y_opt_tensor,
                maximize=self.maximization,
                std_change=self.std_change,
                recalibrator=recalibrator,
            )
        elif self.acquisition == "UCB":
            self.acquisition_function = UpperConfidenceBound(
                self.surrogate_model,
                beta=self.parameters.beta,
                maximize=self.maximization,
                std_change=self.std_change,
                recalibrator=recalibrator,
            )
        elif self.acquisition == "RS":
            self.acquisition_function = RandomSearch()
        elif self.acquisition == "TS":
            self.acquisition_function = MinPosteriorSampling(model=self.surrogate_model, replacement=True, surrogate_type=self.parameters.surrogate)
        else:
            raise ValueError(f"Acquisition function {self.acquisition} not supported.")

        if self.fully_bayes:
            self.acquisition_function = FullyBayes(
                self.surrogate_model,
                self.acquisition_function,
                y_opt_tensor=y_opt_tensor,
            )

    def bo_iter(
        self,
        dataset: Dataset,
        X_pool: np.ndarray = None,
        recalibrator: Recalibrator = None,
        return_idx: bool = False,
    ) -> Dict[np.ndarray, np.ndarray]:
        assert self.is_fitted

        self.construct_acquisition_function(dataset, recalibrator)

        #Why do we sample X_test again here???
        #if X_pool is None:
        #    X_pool, _, _ = dataset.sample_testset(self.n_pool)
        #    idxs = list(range(self.n_pool))
        #    X_pool_entire = X_pool.copy()
#        elif dataset.data.X_test.shape[0] > 1000:
#            idxs = np.random.permutation(dataset.data.X_test.shape[0])[:1000]
#            X_test = dataset.data.X_test[idxs, :]
#            X_test_entire = dataset.data.X_test.copy()
        X_pool = dataset.data.X_pool.copy()
        X_pool_entire = dataset.data.X_pool.copy()
        idxs = list(range(dataset.data.X_pool.shape[0]))
        
        if self.parameters.acquisition == "TS":
            #Using Botorches MaxPosteriorSampling requires dimension to be (1xNxd) instead of (Nx1xd) in other acq. functions.
            X_pool_torch = torch.from_numpy(np.expand_dims(X_pool, 0))

            x_optim = self.acquisition_function(X_pool_torch)
            i_choice = np.random.choice(np.nonzero((x_optim.squeeze(0).detach().numpy()==X_pool_entire).all(axis=1))[0])

            if return_idx:
                return(
                    X_pool_entire[[idxs[i_choice]], :],
                    "N/A",
                    idxs[i_choice],
                )
            else:
                return (
                    X_pool_entire[[idxs[i_choice]], :],
                    "N/A",
                )
        else:
            X_pool_torch = torch.from_numpy(np.expand_dims(X_pool, 1))

            acquisition_values = (
                self.acquisition_function(X_pool_torch.float()).detach().numpy()
            )
            # find idx
            if self.parameters.prob_acq:
                acquisition_values += 1e-8  # numerical adjust.
                p = acquisition_values / np.sum(acquisition_values)
                i_choice = np.random.choice(range(len(p)), p=p)
            else:
                i_choice = np.random.choice(
                    np.flatnonzero(acquisition_values == acquisition_values.max())
                )
            if return_idx:
                return (
                    X_pool_entire[[idxs[i_choice]], :],
                    acquisition_values[i_choice],
                    idxs[i_choice],
                )
            else:
                return (
                    X_pool_entire[[idxs[i_choice]], :],
                    acquisition_values[i_choice],
                )