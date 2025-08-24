from imports.general import *
from imports.ml import *
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import LeavePOut as LeaveKOut
from sklearn.model_selection import KFold, LeaveOneOut
from src.dataset import Dataset
from src.parameters import Parameters
from copy import deepcopy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class RecalibratorUNIBOv2(object):
    
    def __init__(
        self,
        dataset: Dataset,
        model,
        parameters: Parameters,
        mode: str = "cv",
    ) -> None:

        if mode == "cv":
            self.cv_module = LeaveOneOut()
        else:
            raise NotImplementedError("Only CV implemented for this procedure.")
        
        quantile_level = parameters.quantile_level        
        self.mode = mode
        temp_model = deepcopy(model)
        mus, sigmas, ys_true = self.make_recal_dataset(dataset, temp_model)
        self.recalibrator_model = self.train_recalibrator_model(mus, sigmas, ys_true, recalibration_method=parameters.recalibration_method)
        self.recalibrated_z_score, self.scaling_factor = self._calculate_scaling_factor(quantile_level=quantile_level, recalibration_method=parameters.recalibration_method)
        #print("Vanilla z-score:", norm.ppf(quantile_level))
        #print("Recalibrated z-score:", self.recalibrated_z_score)
        
    def make_recal_dataset(self, dataset: Dataset, model):
        X_train, y_train = dataset.data.X_train, dataset.data.y_train
        mus, sigmas, ys_true = np.array([]), np.array([]), np.array([])
        for train_index, val_index in self.cv_module.split(X_train):
            X_train_, y_train_ = X_train[train_index, :], y_train[train_index]
            X_val, y_val = X_train[val_index, :], y_train[val_index]
            model.fit(X_train_, y_train_)
            mus_val, sigs_val = model.predict(X_val)
            mus = np.append(mus, mus_val)
            sigmas = np.append(sigmas, sigs_val)
            ys_true = np.append(ys_true, y_val.squeeze())
        return mus, sigmas, ys_true

    def train_recalibrator_model(self, mu_test, sig_test, y_val, recalibration_method: str ="isotonic"):
        CDF = norm.cdf(y_val.squeeze(), mu_test.squeeze(), sig_test.squeeze()).squeeze()
        P = np.vectorize(lambda p: np.mean(CDF < p))
        P_hat = P(CDF)

        if recalibration_method == "isotonic":
            model = IsotonicRegression(
                y_min=0, y_max=1, increasing=True, out_of_bounds="clip"
            ).fit(CDF, P_hat)
        elif recalibration_method == "gp": 
            kernel = RBF() + WhiteKernel()
            model = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10,
                normalize_y=True,
                random_state=0
            ).fit(CDF.reshape(-1,1), P_hat)
        else: raise ValueError(f"Recalibration method '{recalibration_method}' not recognized.")
        return model    
    
    def _calculate_scaling_factor(self, quantile_level: float, recalibration_method: str = "isotonic"):
        
        if recalibration_method == "isotonic":
            calibrated_ps = self.recalibrator_model.y_thresholds_
            uncalibrated_ps = self.recalibrator_model.X_thresholds_
            p_uncalibrated = np.interp(quantile_level, calibrated_ps, uncalibrated_ps)
        elif recalibration_method == "gp":    
            uncalibrated_ps_grid = np.linspace(0, 1, 500).reshape(-1, 1)
            calibrated_ps_pred = self.recalibrator_model.predict(uncalibrated_ps_grid)
            p_uncalibrated = np.interp(quantile_level, calibrated_ps_pred, uncalibrated_ps_grid.ravel())
            
        p_uncalibrated = np.clip(p_uncalibrated, 0.0, 1.0 - 1e-9)
        z_recalibrated = norm.ppf(p_uncalibrated)
        z_vanilla = norm.ppf(quantile_level)
        
        if np.isclose(z_vanilla, 0):
            return 1.0 # Avoid division by zero
        
        scaling_factor = z_recalibrated / z_vanilla
        return z_recalibrated, scaling_factor

    def recalibrate(self, mu_preds, sig_preds):
        mu_recalibrated = mu_preds        
        sig_recalibrated = sig_preds * self.scaling_factor
        return mu_recalibrated, sig_recalibrated

class RecalibratorUNIBOv1(object):
    def __init__(self, dataset: Dataset, model, parameters: Parameters, mode: str = "cv") -> None:
        if mode == "kfold":
            self.cv_module = KFold(n_splits=parameters.n_initial)
        else:
            self.cv_module = LeaveOneOut()
        self.mode = mode
        
        temp_model = deepcopy(model)
        mus, sigmas, ys_true = self.make_recal_dataset(dataset, temp_model)
        self.recalibrator_model = self.train_recalibrator_model(
            mus, sigmas, ys_true, recalibration_method=parameters.recalibration_method
        )
        test = np.linspace(0.01, 0.99, 50)
        plt.plot(test, self.recalibrator_model.predict(test.reshape(-1, 1)))
        plt.show()
        
    def make_recal_dataset(self, dataset: Dataset, model):
        if self.mode == "cv" or self.mode == "kfold":
            X_train, y_train = dataset.data.X_train, dataset.data.y_train
            mus, sigmas, ys_true = np.array([]), np.array([]), np.array([])
            for train_index, val_index in self.cv_module.split(X_train):
                X_train_, y_train_ = X_train[train_index, :], y_train[train_index]
                X_val, y_val = X_train[val_index, :], y_train[val_index]
                model.fit(X_train_, y_train_)
                mus_val, sigs_val = model.predict(X_val)
                mus = np.append(mus, mus_val)
                sigmas = np.append(sigmas, sigs_val)
                ys_true = np.append(ys_true, y_val.squeeze())

            return mus, sigmas, ys_true
        elif self.mode == "iid":
            X_val, y_val = dataset.data.X_val, dataset.data.y_val
            model.fit(dataset.data.X_train, dataset.data.y_train)
            mus_val, sigs_val = model.predict(X_val)
            return mus_val, sigs_val, y_val
        
    def train_recalibrator_model(self, mu_test, sig_test, y_val, recalibration_method: str = "isotonic"):
        CDF = norm.cdf(y_val.squeeze(), mu_test.squeeze(), sig_test.squeeze()).squeeze()
        P = np.vectorize(lambda p: np.mean(CDF < p))
        P_hat = P(CDF)
        plt.scatter(CDF, P_hat)
        if recalibration_method == "isotonic":
            model = IsotonicRegression(
                y_min=0, y_max=1, increasing=True, out_of_bounds="clip"
            ).fit(CDF, P_hat)
        elif recalibration_method == "gp":
            kernel = RBF() + WhiteKernel()
            model = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10,
                normalize_y=True,
                random_state=0
            ).fit(CDF.reshape(-1, 1), P_hat)
        else:
            raise ValueError(f"Recalibration method '{recalibration_method}' not recognized.")
            
        return model

    def estimate_moments_from_ecdf(self, y_space, cdf_hat):
        y_space = y_space.squeeze()
        cdf_hat = cdf_hat.squeeze()
        pdf_hat = np.diff(cdf_hat)
        m1_hat = np.sum(y_space[1:] * pdf_hat)
        m2_hat = np.sum(y_space[1:] ** 2 * pdf_hat)
        v1_hat = m2_hat - m1_hat ** 2
        if not np.isfinite(m1_hat) or not np.isfinite(np.sqrt(v1_hat)):
            raise ValueError()
        return m1_hat, v1_hat

    def recalibrate(self, mu_preds, sig_preds):
        is_tensor = torch.is_tensor(mu_preds)
        mu_preds = (
            mu_preds.cpu().detach().numpy().squeeze()
            if is_tensor
            else mu_preds.squeeze()
        )
        sig_preds = (
            sig_preds.cpu().detach().numpy().squeeze()
            if is_tensor
            else sig_preds.squeeze()
        )

        n_steps = 100
        mu_new, std_new = [], []
        
        if not isinstance(mu_preds, np.ndarray) or mu_preds.ndim == 0:
            mu_preds = np.array([mu_preds])
            sig_preds = np.array([sig_preds])
        
        for mu_i, std_i in zip(mu_preds, sig_preds):
            y_space = np.linspace(mu_i - 3 * std_i, mu_i + 3 * std_i, n_steps)
            cdf = norm.cdf(y_space, mu_i.squeeze(), std_i.squeeze())
            cdf_hat = self.recalibrator_model.predict(cdf.reshape(-1, 1))
            mu_i_hat, v_i_hat = self.estimate_moments_from_ecdf(y_space, cdf_hat)
            mu_new.append(mu_i_hat)
            std_new.append(np.sqrt(v_i_hat))

        if is_tensor:
            return (
                torch.from_numpy(np.array(mu_new)),
                torch.from_numpy(np.array(std_new)),
            )
        else:
            return np.array(mu_new)[:, np.newaxis], np.array(std_new)[:, np.newaxis]
        
class RecalibratorONLINEv1(object):
    """
    Recalibrates the entire quantile function using the Online Subgradient
    Descent method from Deshpande et al., 2024.
    """
    def __init__(self, dataset: Dataset, model, parameters: Parameters, mode: str = "cv") -> None:
        if mode == "kfold":
            self.cv_module = KFold(n_splits=parameters.n_initial)
        else:
            self.cv_module = LeaveOneOut()
        self.mode = mode
        self.eta = parameters.eta
        temp_model = deepcopy(model)
        mus, sigmas, ys_true = self.make_recal_dataset(dataset, temp_model)
        self.recalibrator_model = self.train_recalibrator_model(
            mus, sigmas, ys_true, recalibration_method=parameters.recalibration_method
        )
        test = np.linspace(0.01, 0.99, 50)
        #plt.plot(test, self.recalibrator_model.predict(test.reshape(-1, 1)))
        #plt.show()

    def make_recal_dataset(self, dataset: Dataset, model):
        if self.mode == "cv" or self.mode == "kfold":
            X_train, y_train = dataset.data.X_train, dataset.data.y_train
            mus, sigmas, ys_true = np.array([]), np.array([]), np.array([])
            for train_index, val_index in self.cv_module.split(X_train):
                X_train_, y_train_ = X_train[train_index, :], y_train[train_index]
                X_val, y_val = X_train[val_index, :], y_train[val_index]
                model.fit(X_train_, y_train_)
                mus_val, sigs_val = model.predict(X_val)
                mus = np.append(mus, mus_val)
                sigmas = np.append(sigmas, sigs_val)
                ys_true = np.append(ys_true, y_val.squeeze())

            return mus, sigmas, ys_true
        elif self.mode == "iid":
            X_val, y_val = dataset.data.X_val, dataset.data.y_val
            model.fit(dataset.data.X_train, dataset.data.y_train)
            mus_val, sigs_val = model.predict(X_val)
            return mus_val, sigs_val, y_val
        
    def train_recalibrator_model(self, mu_test, sig_test, y_val, recalibration_method: str = "isotonic"):
        
        p_vals = np.linspace(0.01, 0.99, 50)
        q_vals = np.array([self._online_quantile_recalibration(mu_test, sig_test, y_val, p, self.eta) for p in p_vals])
        #plt.scatter(p_vals, q_vals, label="Recalibrated Quantiles")
        if recalibration_method == "isotonic":
            model = IsotonicRegression(
                y_min=0, y_max=1, increasing=True, out_of_bounds="clip"
            ).fit(p_vals, q_vals)
        elif recalibration_method == "gp":
            kernel = RBF() + WhiteKernel()
            model = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10,
                normalize_y=True,
                random_state=0
            ).fit(p_vals.reshape(-1, 1), q_vals)
        else:
            raise ValueError(f"Recalibration method '{recalibration_method}' not recognized.")
            
        return model
    
    def _online_quantile_recalibration(self, mus, sigmas, ys_true, p, eta):
        """Runs the OSD update rule to find the recalibrated quantile."""
        q_t = p
        for i in range(len(ys_true)):
            mu_i, sigma_i, y_i = mus[i], sigmas[i], ys_true[i]
            q_val_t = norm.ppf(q_t, loc=mu_i, scale=sigma_i)
            indicator = 1.0 if y_i <= q_val_t else 0.0
            gradient = indicator - p
            q_t = q_t - eta * gradient
            q_t = np.clip(q_t, 1e-3, 1.0 - 1e-3)
        return q_t

    def estimate_moments_from_ecdf(self, y_space, cdf_hat):
        y_space = y_space.squeeze()
        cdf_hat = cdf_hat.squeeze()
        pdf_hat = np.diff(cdf_hat)
        m1_hat = np.sum(y_space[1:] * pdf_hat)
        m2_hat = np.sum(y_space[1:] ** 2 * pdf_hat)
        v1_hat = m2_hat - m1_hat ** 2
        if not np.isfinite(m1_hat) or not np.isfinite(np.sqrt(v1_hat)):
            raise ValueError()
        return m1_hat, v1_hat

    def recalibrate(self, mu_preds, sig_preds):
        is_tensor = torch.is_tensor(mu_preds)
        mu_preds = (
            mu_preds.cpu().detach().numpy().squeeze()
            if is_tensor
            else mu_preds.squeeze()
        )
        sig_preds = (
            sig_preds.cpu().detach().numpy().squeeze()
            if is_tensor
            else sig_preds.squeeze()
        )

        n_steps = 100
        mu_new, std_new = [], []
        
        if not isinstance(mu_preds, np.ndarray) or mu_preds.ndim == 0:
            mu_preds = np.array([mu_preds])
            sig_preds = np.array([sig_preds])
        
        for mu_i, std_i in zip(mu_preds, sig_preds):
            y_space = np.linspace(mu_i - 4 * std_i, mu_i + 4 * std_i, n_steps)
            cdf = norm.cdf(y_space, mu_i.squeeze(), std_i.squeeze())
            cdf_hat = self.recalibrator_model.predict(cdf.reshape(-1, 1))
            mu_i_hat, v_i_hat = self.estimate_moments_from_ecdf(y_space, cdf_hat)
            mu_new.append(mu_i_hat)
            std_new.append(np.sqrt(v_i_hat))

        if is_tensor:
            return (
                torch.from_numpy(np.array(mu_new)),
                torch.from_numpy(np.array(std_new)),
            )
        else:
            return np.array(mu_new)[:, np.newaxis], np.array(std_new)[:, np.newaxis]

class RecalibratorONLINEv2(object):
    """
    Recalibrates a single quantile using the Online Subgradient Descent method
    from Deshpande et al., 2024[cite: 98, 133]. This recalibrated quantile is then
    used to derive a scaling factor for the predictive standard deviation.
    """
    def __init__(self, dataset: Dataset, model, parameters: Parameters, mode: str = "cv") -> None:
        if mode != "cv":
            raise NotImplementedError("Only 'cv' mode with LeaveOneOut is supported.")
        
        self.cv_module = LeaveOneOut()
        
        temp_model = deepcopy(model)
        mus, sigmas, ys_true = self.make_recal_dataset(dataset, temp_model)
        target_quantile = parameters.quantile_level
        learning_rate = parameters.eta
        final_q = self._online_quantile_recalibration(
            mus, sigmas, ys_true, target_quantile, learning_rate
        )
        self.recalibrated_z_score = norm.ppf(final_q)
        vanilla_z_score = norm.ppf(target_quantile)
        #print("Vanilla z-score:", vanilla_z_score)
        #print("Recalibrated z-score:", self.recalibrated_z_score)
        if np.isclose(vanilla_z_score, 0):
            self.scaling_factor = 1.0 # Avoid division by zero
        else:
            self.scaling_factor = self.recalibrated_z_score / vanilla_z_score
        # Prevent drifting to negative values/0 after the calibration set becomes increasingly biased.
        self.scaling_factor = np.maximum(self.scaling_factor, 1e-6)
    def _online_quantile_recalibration(self, mus, sigmas, ys_true, p, eta):
        """Runs the OSD update rule to find the recalibrated quantile[cite: 133]."""
        q_t = p # Start with the target quantile
        
        for i in range(len(ys_true)):
            mu_i, sigma_i, y_i = mus[i], sigmas[i], ys_true[i]            
            q_val_t = norm.ppf(q_t, loc=mu_i, scale=sigma_i)
            indicator = 1.0 if y_i <= q_val_t else 0.0
            gradient = indicator - p
            q_t = q_t - eta * gradient
            q_t = np.clip(q_t, 1e-3, 1.0 - 1e-3)

        #print("Recalibrated quantile:", q_t)
        return q_t

    def make_recal_dataset(self, dataset: Dataset, model):
        X_train, y_train = dataset.data.X_train, dataset.data.y_train
        mus, sigmas, ys_true = np.array([]), np.array([]), np.array([])
        for train_index, val_index in self.cv_module.split(X_train):
            X_train_, y_train_ = X_train[train_index, :], y_train[train_index]
            X_val, y_val = X_train[val_index, :], y_train[val_index]
            model.fit(X_train_, y_train_)
            mus_val, sigs_val = model.predict(X_val)
            mus = np.append(mus, mus_val)
            sigmas = np.append(sigmas, sigs_val)
            ys_true = np.append(ys_true, y_val.squeeze())
        return mus, sigmas, ys_true

    def recalibrate(self, mu_preds, sig_preds):
        """Applies the single learned scaling factor to the standard deviation."""
        return mu_preds, sig_preds * self.scaling_factor