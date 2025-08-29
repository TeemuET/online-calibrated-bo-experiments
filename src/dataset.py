from argparse import ArgumentError
from typing import Dict
from src.parameters import *
from imports.general import *
from datasets.benchmarks.benchmark import Benchmark

class Dataset(object):
    """Handles dataset loading, preprocessing, and splitting."""
    def __init__(self, parameters: Parameters) -> None:
        self.__dict__.update(asdict(parameters))
        if parameters.data_name.lower() == "benchmark":
            self.data = Benchmark(parameters)
        else:
            raise ArgumentError(f"{parameters.problem} problem in parameters not found")

        self.summary = {
            "data_name": self.data_name,
            "x_ubs": self.data.x_ubs.tolist(),
            "x_lbs": self.data.x_lbs.tolist(),
            "X_mean_scaler": self.data.X_mean_pool_scaling.tolist(),
            "y_mean_scaler": self.data.y_mean_pool_scaling.tolist(),
            "X_mean": self.data.X_mean_pool.tolist(),
            "y_mean": self.data.y_mean_pool.tolist(),
            "X_std": self.data.X_std_pool.tolist(),
            "y_std": self.data.y_std_pool.tolist(),
            "X_test": self.data.X_test.tolist(),
            "y_test": self.data.y_test.tolist(),
            "X_train": self.data.X_train.tolist(),
            "y_train": self.data.y_train.tolist(),
            "X_pool": self.data.X_pool.tolist(),
            "y_pool": self.data.y_pool.tolist(),
            "y_min_idx_pool": float(self.data.y_min_idx_pool),
            "y_min_loc_pool": self.data.y_min_loc_pool.tolist(),
            "y_min_pool": float(self.data.y_min_pool),
            "y_min_idx_test": float(self.data.y_min_idx_test),
            "y_min_loc_test": self.data.y_min_loc_test.tolist(),
            "y_min_test": float(self.data.y_min_test),
        }
        if not self.data.real_world:
            self.summary.update(
                {
                    "signal_std": float(self.data.signal_std_pool),
                    "noise_std": float(self.data.noise_std),
                    "f_min_pool": float(self.data.f_min_pool),
                    "f_min_test": float(self.data.f_min_test),
                    "ne_true": float(self.data.ne_true),
                    "f_min_loc_pool": self.data.f_min_loc_pool.tolist(),
                    "f_min_idx_pool": float(self.data.f_min_idx_pool),
                    "f_min_loc_test": self.data.f_min_loc_test.tolist(),
                    "f_min_idx_test": float(self.data.f_min_idx_test),
                    "f_test": self.data.f_test.tolist(),
                    "f_train": self.data.f_train.tolist(),
                }
            )

        self.actual_improvement = None
        self.expected_improvement = None
        self.update_solution()

    def update_solution(self) -> None:
        self.opt_idx = (
            np.argmax(self.data.y_train)
            if self.maximization
            else np.argmin(self.data.y_train)
        )
        self.y_opt = self.data.y_train[[self.opt_idx], :]
        self.X_y_opt = self.data.X_train[[self.opt_idx], :]

        self.summary.update(
            {
                "X_train": self.data.X_train.tolist(),
                "y_train": self.data.y_train.tolist(),
                "opt_idx": int(self.opt_idx),
                "X_y_opt": self.X_y_opt.tolist(),
                "y_opt": self.y_opt.tolist(),
            }
        )
        if not self.data.real_world:
            self.f_opt = self.data.f_train[[self.opt_idx], :]
            self.X_f_opt = self.data.X_train[[self.opt_idx], :]
            self.summary.update(
                {
                    "X_f_opt": self.X_f_opt.tolist(),
                    "opt_idx": int(self.opt_idx),
                    "f_opt": int(self.f_opt),
                    "f_train": self.data.f_train.tolist()
                }
            )

    def save(self, save_settings: str = "") -> None:
        json_dump = json.dumps(self.summary)
        with open(self.savepth + f"dataset{save_settings}.json", "w") as f:
            f.write(json_dump)

    def add_data(
        self,
        x_new: np.ndarray,
        y_new: np.ndarray,
        f_new: np.ndarray,
        i_choice: int = None,
        acq_val: np.ndarray = None,
    ) -> None:
        self.data.X_train = np.append(self.data.X_train, x_new, axis=0)
        self.data.y_train = np.append(self.data.y_train, y_new, axis=0)
        if not self.data.real_world:
            self.data.f_train = np.append(self.data.f_train, f_new, axis=0)

        if i_choice is not None:
            self.data.X_pool = np.delete(self.data.X_pool, i_choice, axis=0)
            self.data.y_pool = np.delete(self.data.y_pool, i_choice, axis=0)
            if not self.data.real_world:
                self.data.f_pool = np.delete(self.data.f_pool, i_choice, axis=0)

        self.actual_improvement = y_new - self.y_opt if self.bo else None
        self.expected_improvement = acq_val
        self.update_solution()

    def sample_testset(self, n_samples: int = None) -> Dict[np.ndarray, np.ndarray]:
        n_samples = self.n_test if n_samples is None else n_samples
        if self.data.real_world:
            X, y = self.data.sample_data(n_samples=n_samples)
            return X, y
        else:
            X, y, f = self.data.sample_data(n_samples=n_samples)
            return X, y, f
