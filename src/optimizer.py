from src.dataset import Dataset
from src.parameters import *
from surrogates.gaussian_process import GaussianProcess
from botorch.generation.sampling import MaxPosteriorSampling
from typing import Any
from botorch.optim import optimize_acqf
from acquisitions.botorch_acqs import (
    ExpectedImprovement,
    NumericalExpectedImprovement,
    UpperConfidenceBound,
)

import warnings


class Optimizer(object):
    """
    This class handles the Bayesian optimization iteration, including fitting the surrogate model,
    constructing the acquisition function, and querying the optimal points.
    """

    def __init__(self, parameters: Parameters) -> None:
        self.__dict__.update(asdict(parameters))
        self.parameters = parameters
        self.is_fitted = False
        
    def fit_surrogate(self, dataset: Dataset) -> None:
        if self.surrogate == "GP":
            self.surrogate_object = GaussianProcess(self.parameters, dataset)
            self.surrogate_model = self.surrogate_object.model
        elif self.surrogate == "RS":
            self.surrogate_object = None
            self.surrogate_model = None
        else:
            raise ValueError(f"Surrogate function {self.surrogate} not supported.")
        self.is_fitted = True

    def construct_acquisition_function(
        self, dataset: Dataset, recalibrator: Any = None
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
                recalibrator_type=self.parameters.recalibrator_type,
            )
        elif self.acquisition == "UCB":
            self.acquisition_function = UpperConfidenceBound(
                self.surrogate_model,
                beta=self.parameters.beta,
                quantile_level=self.parameters.quantile_level,
                maximize=self.maximization,
                std_change=self.std_change,
                recalibrator=recalibrator,
                recalibrator_type=self.parameters.recalibrator_type,
            )
        else:
            raise ValueError(f"Acquisition function {self.acquisition} not supported.")

    def bo_iter(
        self,
        dataset: Dataset,
        X_pool: np.ndarray = None,
        recalibrator: Any = None,
        return_idx: bool = False,
    ) -> Dict[np.ndarray, np.ndarray]:
        assert self.is_fitted
        
        device = torch.device(self.parameters.device)
        
        self.construct_acquisition_function(dataset, recalibrator)

        X_pool = dataset.data.X_pool.copy()
        X_pool_entire = dataset.data.X_pool.copy()
        idxs = list(range(dataset.data.X_pool.shape[0]))
        
        X_pool_torch = torch.from_numpy(np.expand_dims(X_pool, 1)).to(device)

        acquisition_values = (
            self.acquisition_function(X_pool_torch.float()).cpu().detach().numpy()
        )

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