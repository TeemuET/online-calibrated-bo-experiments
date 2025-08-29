from dataclasses import asdict
import json
from netrc import netrc
import numpy as np
import torch
from numpy import save
from uncertainty_toolbox.metrics import *
from typing import Any
from imports.general import *
from imports.ml import *
from src.parameters import Parameters
from src.dataset import Dataset


class Metrics(object):
    """
    Handles the calculation and storage of all performance metrics.

    This class is responsible for tracking various aspects of the Bayesian
    Optimization experiment, including model calibration, performance,
    and regret. The results are stored in a summary dictionary and can be
    saved to a JSON file.

    Attributes:
        summary (dict): A dictionary to store the history of all calculated
                        metrics over the course of the experiment.
    """

    def __init__(self, parameters: Parameters) -> None:
        self.__dict__.update(asdict(parameters))
        self.n_init = parameters.n_initial
        self.p_array = np.linspace(0.001, 0.999, self.n_calibration_bins)
        self.online_cdf_values = []
        self.online_crps_scores = []
        self.summary = {
            "p_array": self.p_array.tolist(),
            "mean_sharpness": [],
            "sharpness_error_true_minus_model": [],
            "posterior_variance": [],
            "bias_mse": [],
            "bias_nmse": [],
            "f_calibration": [],
            "f_calibration_mse": [],
            "f_calibration_nmse": [],
            "y_calibration": [],
            "y_calibration_mse": [],
            "y_calibration_nmse": [],
            "calibration_local_dist_to_nearest_train_sample": [],
            "calibration_local_y": [],
            "next_sample_train_distance": [],
            "elpd": [],
            "expected_improvement": [],
            "actual_improvement": [],
            "mse": [],
            "nmse": [],
            "y_regret_pool": [],
            "y_regret_test": [],
            "f_regret_pool": [],
            "f_regret_test": [],
            "x_y_opt_dist_pool": [],
            "online_crps": [],
            "crps_test": [],
            "calibration_optimum_dist_bins": [],
            "calibration_optimum_local_y_mse": [],
            "online_calibration_nmse": [],
            "online_calibration_mse": [],
            "x_f_opt_dist_pool": [],
            "x_y_opt_dist_test": [],
            "x_f_opt_dist_test": [],
            "uct_calibration": [],
            "uct_sharpness": [],
        }
        
    # ========================================================================
    # Core Methods
    # ========================================================================

    def save(self, save_settings: str = "") -> None:
        json_dump = json.dumps(self.summary)
        with open(self.savepth + f"metrics{save_settings}.json", "w") as f:
            f.write(json_dump)

    def update_summary(self, update: Dict) -> None:
        for k, v in update.items():
            lst = self.summary[k]
            v = v.tolist() if isinstance(v, np.ndarray) else v
            lst.append(v)
            self.summary.update({k: lst})

    # ========================================================================
    # Sharpness metrics
    # ======================================================================

    def sharpness_gaussian(
        self, dataset: Dataset, mus: np.ndarray, sigmas: np.ndarray
    ) -> None:
        """Calculates the sharpness (negative entropy) of the gaussian distributions 
        with means: mus and standard deviation: sigmas
        """
        sharpness = np.array(
            [-norm.entropy(mus[i], sigmas[i]) for i in range(mus.shape[0])]
        )
        mean_sharpness = np.mean(sharpness)
        self.update_summary(
            {
                "mean_sharpness": mean_sharpness,
                "posterior_variance": np.mean(sigmas) ** 2,
            }
        )
        if (
            not dataset.data.real_world
            and hasattr(dataset.data, "ne_true")
            and dataset.data.ne_true is not None
        ):
            self.update_summary(
                {
                    "sharpness_error_true_minus_model": dataset.data.ne_true
                    - mean_sharpness,
                }
            )

    def sharpness_histogram(
        self, model: Model, X: np.ndarray, n_bins: int = 20
    ) -> None:
        """
        NOT USED
        Calculates the sharpness (negative entropy) of the histogram distributions 
        calculated from input X
        """
        if hasattr(model, "histogram_sharpness"):
            hist_sharpness, mean_hist_sharpness = model.histogram_sharpness(
                X, n_bins=n_bins
            )
            self.summary.update(
                {
                    # f"{model.name}_hist_sharpness": hist_sharpness,
                    f"{model.name}_mean_hist_sharpness": mean_hist_sharpness,
                }
            )

    # ========================================================================
    # Calibration Metrics
    # ========================================================================

    def calibration_f(self, mus: np.ndarray, sigmas: np.ndarray, f: np.ndarray) -> None:
        calibrations = np.full((self.n_calibration_bins,), np.nan)
        assert mus.size == sigmas.size == f.size
        for i_p, p in enumerate(self.p_array):
            fractiles = [norm.ppf(0.5 - p / 2, loc=0, scale=sig) for sig in sigmas]
            lb_indicators = mus + fractiles < f
            ub_indicators = f < mus - fractiles
            indicators = np.logical_and(lb_indicators, ub_indicators)
            calibrations[i_p] = np.mean(indicators)

        self.update_summary(
            {
                "f_calibration": calibrations,
                "f_calibration_mse": np.mean((self.p_array - calibrations) ** 2),
            }
        )

    def calibration_f_batched(
        self, mus: np.ndarray, sigmas: np.ndarray, f: np.ndarray
    ) -> None:
        f = np.tile(f, self.n_calibration_bins)
        p_array_ = np.tile(self.p_array[:, np.newaxis], sigmas.size)
        norms = tdist.Normal(
            torch.tensor(np.zeros(sigmas.size)), torch.tensor(sigmas.squeeze())
        )
        fractiles = norms.icdf(torch.tensor(0.5 - p_array_ / 2))
        f_tensor = torch.tensor(f)
        mus_tensor = torch.tensor(mus.squeeze())
        calibrations = (
            torch.mean(
                (
                    torch.logical_and(
                        mus_tensor + fractiles < f_tensor.T,
                        f_tensor.T < mus_tensor - fractiles,
                    )
                ).float(),
                dim=1,
            )
            .cpu()
            .numpy()
        )

        self.update_summary(
            {
                "f_calibration": calibrations,
                "f_calibration_mse": np.mean((self.p_array - calibrations) ** 2),
                "f_calibration_nmse": np.mean((self.p_array - calibrations) ** 2)
                / np.var(self.p_array),
            }
        )

    def calibration_y_batched(
        self,
        mus: np.ndarray,
        sigmas: np.ndarray,
        y: np.ndarray,
        return_mse: bool = False,
        plot: bool = False,
    ) -> None:
        y_ = np.tile(y, self.n_calibration_bins)
        p_array_ = np.tile(self.p_array[:, np.newaxis], sigmas.size)
        norms = tdist.Normal(
            torch.tensor(mus.squeeze()), torch.tensor(sigmas.squeeze())
        )
        icdfs = norms.icdf(torch.tensor(p_array_))
        calibrations = (
            torch.mean((torch.tensor(y_).T <= icdfs).float(), dim=1).cpu().numpy()
        )

        if plot:
            fig = plt.figure()
            plt.plot(self.p_array, calibrations)
            plt.plot(self.p_array, self.p_array, "--")

        if return_mse:
            return np.nanmean((calibrations - self.p_array) ** 2)
        else:
            self.update_summary(
                {
                    "y_calibration": calibrations,
                    "y_calibration_mse": np.nanmean((calibrations - self.p_array) ** 2),
                    "y_calibration_nmse": np.nanmean((calibrations - self.p_array) ** 2)
                    / np.var(self.p_array),
                }
            )

    def calibration_y(
        self,
        mus: np.ndarray,
        sigmas: np.ndarray,
        y: np.ndarray,
        return_mse: bool = False,
    ) -> None:
        """Calculates the calibration of the target (y).
        ### eq. (3) in "Accurate Uncertainties for Deep Learning Using Calibrated Regression"
        """
        p_array = np.linspace(0, 1, self.n_calibration_bins)
        calibrations = np.full((self.n_calibration_bins,), np.nan)
        for i_p, p in enumerate(p_array):
            indicators = y <= [
                norm.ppf(p, loc=mu, scale=sig) for mu, sig in zip(mus, sigmas)
            ]
            calibrations[i_p] = np.mean(indicators)

        if return_mse:
            return np.nanmean((calibrations - p_array) ** 2)
        else:
            self.update_summary(
                {
                    "y_calibration": calibrations,
                    "y_calibration_mse": np.nanmean((calibrations - p_array) ** 2),
                    "y_calibration_nmse": np.nanmean((calibrations - p_array) ** 2)
                    / np.var(p_array),
                }
            )

    def calibration_y_local(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        mus: np.ndarray,
        sigmas: np.ndarray,
        n_bins: int = 20,
    ) -> None:
        """Calculates the calibration of the target (y).
        # eq. (3) in "Accurate Uncertainties for Deep Learning Using Calibrated Regression"
        """

        pair_dists = cdist(X_train, X_test, metric="euclidean")
        pair_dists = np.min(
            pair_dists, axis=0
        )  # only take radius of nearest training point
        counts, bins = np.histogram(
            pair_dists.flatten(), bins=n_bins
        )  # get histogram with n_bins
        calibrations_intervals = np.full((n_bins,), np.nan)
        calibrations = np.full((n_bins,), np.nan)
        for i in range(len(bins) - 1):
            cond = np.logical_and(bins[i] <= pair_dists, pair_dists <= bins[i + 1])
            if np.sum(cond) > 0:
                mus_, sigmas_, y_ = mus[cond], sigmas[cond], y_test[cond]
                calibrations_intervals[i] = self.calibration_y_batched(
                    mus_, sigmas_, y_, return_mse=True
                )
                calibrations[i] = np.nansum(
                    counts[: i + 1]
                    / np.sum(counts[: i + 1])
                    * calibrations_intervals[: i + 1]
                )
        self.update_summary(
            {
                "calibration_local_dist_to_nearest_train_sample": bins[1:],
                "calibration_local_y": calibrations_intervals,
            }
        )

    def calibration_y_local_optimum(
        self,
        dataset: Dataset,
        X_test: np.ndarray,
        y_test: np.ndarray,
        mus: np.ndarray,
        sigmas: np.ndarray,
        n_bins: int = 20,
    ) -> None:

        optimum_loc = np.atleast_2d(dataset.data.y_min_loc_pool)
        distances = cdist(optimum_loc, X_test, metric="euclidean").flatten()
        
        max_dist = np.max(distances) if distances.size > 0 else 1.0
        counts, bin_edges = np.histogram(distances, bins=n_bins, range=(0, max_dist))
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        calibration_errors = np.full((n_bins,), np.nan)

        for i in range(n_bins):
            lower_bound, upper_bound = bin_edges[i], bin_edges[i+1]
            in_bin_selector = (distances >= lower_bound) & (distances <= upper_bound) if i == n_bins - 1 \
                              else (distances >= lower_bound) & (distances < upper_bound)

            if np.any(in_bin_selector):
                mus_bin, sigmas_bin, y_bin = mus[in_bin_selector], sigmas[in_bin_selector], y_test[in_bin_selector]

                error = self.calibration_y_batched(
                    mus_bin, sigmas_bin, y_bin, return_mse=True
                )
                calibration_errors[i] = error

        self.update_summary({
            "calibration_optimum_dist_bins": bin_centers.tolist(),
            "calibration_optimum_local_y_mse": calibration_errors.tolist(),
        })

    def update_online_calibration_data(self, surrogate: Model, recalibrator: Any, x_next: np.ndarray, y_next: np.ndarray) -> None:
        """
        Calculates the model's predicted CDF for the next observation and stores it.
        This must be called BEFORE the model is updated with (x_next, y_next).
        """
        mu_next, sigma_next = surrogate.predict(x_next)
        
        if recalibrator is not None:
            mu_next, sigma_next = recalibrator.recalibrate(mu_next, sigma_next)
        
        sigma_next = np.maximum(sigma_next, 1e-9)
        cdf_val = norm.cdf(y_next, loc=mu_next, scale=sigma_next)
        self.online_cdf_values.append(cdf_val.squeeze())
        
    def calculate_online_calibration(self) -> None:
        """
        Calculates the online calibration score based on all sequentially collected data.
        This implements the logic for cal = Σ(pj - ˆpj)².
        """
        if not self.online_cdf_values:
            return

        observed_cdfs = np.array(self.online_cdf_values)
        p_hat_array = np.mean((observed_cdfs[:, np.newaxis] <= self.p_array), axis=0)

        mse = np.mean((p_hat_array - self.p_array) ** 2)
        
        p_array_var = np.var(self.p_array)
        nmse = mse / p_array_var if p_array_var > 0 else 0.0

        self.update_summary({
            "online_calibration_mse": mse,
            "online_calibration_nmse": nmse,
        })

    # ========================================================================
    # Proper Scores
    # ========================================================================
    
    def expected_log_predictive_density(
        self, mus: np.ndarray, sigmas: np.ndarray, y: np.ndarray,
    ) -> None:
        """Calculates expected log predictive density (elpd) using
        \mathbb{E}\left[\log p_\theta(\textbf{y}|\textbf{X})\right]  
        which essientially is "on average how likely is a new test data under the model".
        """
        log_pdfs = np.array(
            [
                norm.logpdf(y[i], loc=mus[i], scale=sigmas[i])
                for i in range(sigmas.shape[0])
            ]
        )
        elpd = np.mean(log_pdfs)
        self.update_summary({"elpd": elpd})

    def calculate_crps_test_set(
        self, mus: np.ndarray, sigmas: np.ndarray, y: np.ndarray
    ) -> None:
        """
        Calculates the CRPS on a large, fixed test set.
        """
        
        crps_scores = crps_gaussian(mus.flatten(), sigmas.flatten(), y.flatten())
        mean_crps = np.mean(crps_scores)
        self.update_summary({"crps_test": mean_crps})


    def update_online_crps(self, surrogate: Model, recalibrator: Any, x_next: np.ndarray, y_next: np.ndarray) -> None:
        """
        Calculates CRPS for the single next observation and updates the running list.
        This must be called BEFORE the model is updated with (x_next, y_next).
        """
        mu_next, sigma_next = surrogate.predict(x_next)
        
        if recalibrator is not None:
            mu_next, sigma_next = recalibrator.recalibrate(mu_next, sigma_next)
        
        single_crps = crps_gaussian(mu_next.flatten(), sigma_next.flatten(), y_next.flatten())
        
        # Store the individual score
        self.online_crps_scores.append(single_crps.squeeze())
        
        # Update the summary with the running average
        self.update_summary({"online_crps": np.mean(self.online_crps_scores)})

    # ========================================================================
    # BO performance metrics
    # ========================================================================

    def improvement(self, dataset: Dataset):
        self.update_summary(
            {
                "expected_improvement": np.array(dataset.expected_improvement),
                "actual_improvement": np.array(dataset.actual_improvement),
            }
        )

    def bias(self, mus: np.ndarray, f: np.ndarray) -> None:
        mse = np.mean((mus - f) ** 2)
        nmse = mse / np.var(f)
        self.update_summary(
            {"bias_mse": mse, "bias_nmse": nmse,}
        )

    def nmse(self, y: np.ndarray, predictions: np.ndarray) -> None:
        """Calculates normalized mean square error by 
        nmse = \ frac{1}{N\cdot\mathbb{V}[\textbf{y}]} \sum_i (\textbf{y}-\hat{\textbf{y}})^2
        where N is the length of y
        """
        mse = np.mean((y - predictions) ** 2)
        nmse = mse / np.var(y)
        self.update_summary({"mse": mse, "nmse": nmse})

    def regret(self, dataset: Dataset) -> None:
        y_regret_pool = np.abs(dataset.data.y_min_pool.squeeze() - dataset.y_opt.squeeze())
        y_regret_test = np.abs(dataset.data.y_min_test.squeeze() - dataset.y_opt.squeeze())
        self.update_summary({"y_regret_pool": y_regret_pool.squeeze()})
        self.update_summary({"y_regret_test": y_regret_test.squeeze()})
        if not dataset.data.real_world:
            f_regret_pool = np.abs(dataset.data.f_min_pool - dataset.f_opt)
            f_regret_test = np.abs(dataset.data.f_min_test - dataset.f_opt)
            self.update_summary({"f_regret_pool": f_regret_pool.squeeze()})
            self.update_summary({"f_regret_test": f_regret_test.squeeze()})

    def glob_min_dist(self, dataset: Dataset, test_set: bool =False) -> None:
        if not test_set:
            y_squared_error = (dataset.X_y_opt - np.array(dataset.data.y_min_loc_pool)) ** 2
            self.update_summary(
                {"x_y_opt_dist_pool": np.sqrt(np.sum(y_squared_error)),}
            )
            if not dataset.data.real_world:
                f_squared_error = (dataset.X_f_opt - np.array(dataset.data.f_min_loc_pool)) ** 2
                self.update_summary(
                    {"x_f_opt_dist_pool": np.sqrt(np.sum(f_squared_error)),}
                )
        else:
            y_squared_error = (dataset.X_y_opt - np.array(dataset.data.y_min_loc_test)) ** 2
            self.update_summary(
                {"x_y_opt_dist_test": np.sqrt(np.sum(y_squared_error)),}
            )
            if not dataset.data.real_world:
                f_squared_error = (dataset.X_f_opt - np.array(dataset.data.f_min_loc_test)) ** 2
                self.update_summary(
                    {"x_f_opt_dist_test": np.sqrt(np.sum(f_squared_error)),}
                )

    # ========================================================================
    # Uncretainty Toolbox Metrics and exploration metric
    # ======================================================================
    def run_uct(self, mu_test, sigma_test, y_test):
        uct_metrics = uct.metrics.get_all_metrics(
            mu_test.squeeze(), sigma_test.squeeze(), y_test.squeeze(), verbose=False,
        )
        self.update_summary(
            {
                "uct_calibration": uct_metrics["avg_calibration"]["rms_cal"],
                "uct_sharpness": uct_metrics["sharpness"]["sharp"],
            }
        )

    def calculate_exploration(self, dataset: Dataset):
        dists = cdist(dataset.data.X_train[:-2], [dataset.data.X_train[-1]], metric='euclidean')
        min_dist = np.min(dists, axis=0)
        self.update_summary(
            {
                "next_sample_train_distance": min_dist,
            }
        )

    def analyze(
        self,
        surrogate: Model,
        dataset: Dataset,
        recalibrator: Any = None,
        extensive: bool = True,
    ) -> None:
        if surrogate is not None and extensive:
            
            X_test = dataset.data.X_test
            y_test = dataset.data.y_test
            if not dataset.data.real_world:
                f_test = dataset.data.f_test

            mu_test, sigma_test = surrogate.predict(X_test)
            if recalibrator is not None:
                mu_test, sigma_test = recalibrator.recalibrate(mu_test, sigma_test)
            if not dataset.data.real_world:
                self.calibration_f_batched(mu_test, sigma_test, f_test)
            self.calibration_y_batched(mu_test, sigma_test, y_test)
            self.calibration_y_local(
                dataset.data.X_train, X_test, y_test, mu_test, sigma_test
            )
            self.calibration_y_local_optimum(
            dataset, X_test, y_test, mu_test, sigma_test
            )
            self.calculate_crps_test_set(mu_test, sigma_test, y_test)
            self.sharpness_gaussian(dataset, mu_test, sigma_test)
            self.expected_log_predictive_density(
                mu_test, sigma_test, y_test,
            )
            self.nmse(y_test, mu_test)
            if not dataset.data.real_world:
                self.bias(mu_test, f_test)
            self.run_uct(mu_test, sigma_test, y_test)
            self.improvement(dataset)
            self.regret(dataset)
            self.glob_min_dist(dataset, test_set=True)
            self.glob_min_dist(dataset, test_set=False)
            if dataset.data.X_train.shape[0] > self.n_init:
                self.calculate_exploration(dataset)

