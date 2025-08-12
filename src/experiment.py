from dataclasses import asdict
from imports.general import *
from imports.ml import *
from src.metrics import Metrics
from src.dataset import Dataset
from src.optimizer import Optimizer
from .parameters import Parameters
from src.recalibrator import Recalibrator


class Experiment(object):
    def __init__(self, parameters: Parameters) -> None:
        self.__dict__.update(asdict(parameters))
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.dataset = Dataset(parameters)
        self.optimizer = Optimizer(parameters)
        self.metrics = Metrics(parameters)
        self.parameters = parameters

    def __str__(self):
        return (
            "Experiment:"
            + self.dataset.__str__
            + "\r\n"
            + self.optimizer.__str__
            + "\r\n"
            + self.metrics.__str__
        )


    def _track_surrogate_state(self, epoch: Union[int, str], surrogate: object) -> None:
        """Prints the current state of the surrogate model for tracking."""
        if not self.parameters.track_surrogate_state:
            return
        header = f"--- Epoch {epoch+1}/{self.n_evals} ---" if isinstance(epoch, int) else f"--- {epoch} State ---"
        print(f"\n{header}")
        print(f"Surrogate type: {type(surrogate).__name__}")

        # For Gaussian Process, print hyperparameters
        if hasattr(surrogate, "model") and hasattr(surrogate.model, "covar_module"):
            try:
                lengthscale = surrogate.model.covar_module.base_kernel.lengthscale.item()
                outputscale = surrogate.model.covar_module.outputscale.item()
                noise = surrogate.model.likelihood.noise.item()
                print(f"  GP Lengthscale: {lengthscale:.4f}")
                print(f"  GP Outputscale: {outputscale:.4f}")
                print(f"  GP Likelihood Noise: {noise:.4f}")
            except AttributeError:
                print("  Could not retrieve all GP hyperparameters.")
        
        # For Bayesian Neural Network
        elif "BNN" in type(surrogate).__name__:
            num_params = sum(p.numel() for p in surrogate.model.parameters() if p.requires_grad)
            print(f"  BNN has {num_params} trainable parameters.")

        print("--------------------------")

    def run(self) -> None:

        # Epoch 0
        self.optimizer.fit_surrogate(self.dataset)
        self._track_surrogate_state("Initial", self.optimizer.surrogate_object)
        recalibrator = (
            Recalibrator(
                self.dataset, self.optimizer.surrogate_object, mode=self.recal_mode,
            )
            if self.recalibrate
            else None
        )
        self._track_surrogate_state("Post-recalibration", self.optimizer.surrogate_object)
        self.metrics.analyze(
            self.optimizer.surrogate_object,
            self.dataset,
            recalibrator=recalibrator,
            extensive=True,
        )
        if self.bo:
            # Epochs > 0
            for e in tqdm(range(self.n_evals), desc="BO Iterations"):

                recalibrator = (
                    Recalibrator(
                        self.dataset,
                        self.optimizer.surrogate_object,
                        mode=self.recal_mode,
                    )
                    if self.recalibrate
                    else None
                )
                    
                self._track_surrogate_state(("Post recalibration", e), self.optimizer.surrogate_object)
                if self.parameters.fix_surrogate_logic:
                    self.optimizer.fit_surrogate(self.dataset)
                    
                self._track_surrogate_state(("Before BO iteration", e), self.optimizer.surrogate_object)
                # BO iteration
                x_next, acq_val, i_choice = self.optimizer.bo_iter(
                    self.dataset,
                    X_pool=self.dataset.data.X_pool,
                    recalibrator=recalibrator,
                    return_idx=True,
                )

                y_next = self.dataset.data.y_pool[[i_choice]]
                f_next = (
                    self.dataset.data.f_pool[[i_choice]]
                    if not self.dataset.data.real_world
                    else None
                )

                # add to dataset
                self.dataset.add_data(x_next, y_next, f_next, i_choice=i_choice)

                # Update dataset
                self.dataset.update_solution()

                # Update surrogate
                self.optimizer.fit_surrogate(self.dataset)
                
                self._track_surrogate_state(("After fitting on the new x_next", e), self.optimizer.surrogate_object)
                if self.analyze_all_epochs:
                    self.metrics.analyze(
                        self.optimizer.surrogate_object,
                        self.dataset,
                        recalibrator=recalibrator,
                        extensive=self.extensive_metrics or e == self.n_evals - 1,
                    )

            if not self.analyze_all_epochs:
                self.metrics.analyze(
                    self.optimizer.surrogate_object,
                    self.dataset,
                    recalibrator=recalibrator,
                    extensive=True,
                )
        else:
            if self.analyze_all_epochs:
                for e in tqdm(range(self.n_evals), leave=False):
                    X, y, f = self.dataset.data.sample_data(n_samples=1)
                    self.dataset.add_data(X, y, f)
                    self.optimizer.fit_surrogate(self.dataset)
                    recalibrator = (
                        Recalibrator(
                            self.dataset,
                            self.optimizer.surrogate_object,
                            mode=self.recal_mode,
                        )
                        if self.recalibrate
                        else None
                    )
                    self.metrics.analyze(
                        self.optimizer.surrogate_object,
                        self.dataset,
                        recalibrator=recalibrator,
                        extensive=self.extensive_metrics or e == self.n_evals - 1,
                    )
            else:
                X, y, f = self.dataset.data.sample_data(self.n_evals)
                self.dataset.add_data(X, y, f)
                self.optimizer.fit_surrogate(self.dataset)
                recalibrator = (
                    Recalibrator(
                        self.dataset,
                        self.optimizer.surrogate_object,
                        mode=self.recal_mode,
                    )
                    if self.recalibrate
                    else None
                )
            self.metrics.analyze(
                self.optimizer.surrogate_object,
                self.dataset,
                recalibrator=recalibrator,
                extensive=True,
            )

        self.dataset.save()
        self.metrics.save()


if __name__ == "__main__":
    Experiment()
