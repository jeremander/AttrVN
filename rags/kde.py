import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from typing import Any, Dict, Optional


class TwoClassKDE(BaseEstimator):
    """Class for Kernel Density Estimator on two labels. Likelihood ratio at a point is ratio of class-1 likelihood estimate to class-0 likelihood estimate, times the class odds, where this is calculated as the posterior mean estimate under Beta(1, 1) prior, given the observations. If no points are observed for one of the classes, a default (improper) uniform prior is assumed for that class. """
    def __init__(self, **kwargs: Any):
        """Takes same parameters as KernelDensity estimator."""
        self.kde0 = KernelDensity(**kwargs)
        self.kde1 = KernelDensity(**kwargs)
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits KDE models on the data. X is array of data points, y is array of 0-1 labels."""
        y = np.asarray(y, dtype = int)
        self.n0, self.n1 = (y == 0).sum(), (y == 1).sum()
        assert (self.n0 + self.n1 == len(y)), "y must be vector of 1's and 0's."
        X0, X1 = X[y == 0], X[y == 1]
        if (self.n0 > 0):
            self.kde0.fit(X0)
        if (self.n1 > 0):
            self.kde1.fit(X1)
    def fit_with_optimal_bandwidth(self, X: np.ndarray, y: np.ndarray, gridsize: int = 101, dynamic_range: float = 100, cv: int = 10, verbose: int = 0, n_jobs: int = 1) -> None:
        """Determines optimal bandwidth using the following strategy: For each subset (0 or 1) of the dataset, 1) set b = 1.06 * sigma * n^(-1/5), the Silverman's rule of thumb estimate for the optimal bandwidth. sigma is the sample standard deviation of the samples after zero-centering the columns (note: ideally each column will have comparable variance), 2) set up a grid (of size gridsize) of bandwidth values to try, ranging from b / alpha to b * alpha in geometric progression, where alpha = sqrt(dynamic_range), 3) compute average likelihood of the estimator on the data using cv-fold cross-validation, 4) select the bandwidth with the highest likelihood."""
        y = np.asarray(y, dtype = int)
        self.n0, self.n1 = (y == 0).sum(), (y == 1).sum()
        assert (self.n0 + self.n1 == len(y)), "y must be vector of 1's and 0's."
        X0, X1 = X[y == 0], X[y == 1]
        if (self.n0 > 0):
            log_b0 = np.log(1.06) + np.log((X0 - X0.mean(axis = 0)).std()) - 0.2 * np.log(self.n0)
            grid0 = GridSearchCV(self.kde0, {'bandwidth' : np.exp(np.linspace(log_b0 - 0.5 * np.log(dynamic_range), log_b0 + 0.5 * np.log(dynamic_range), gridsize))}, cv = cv, verbose = verbose, n_jobs = n_jobs)
            grid0.fit(X0)
            self.kde0 = grid0.best_estimator_
        if (self.n1 > 0):
            log_b1 = np.log(1.06) + np.log((X1 - X1.mean(axis = 0)).std()) - 0.2 * np.log(self.n1)
            grid1 = GridSearchCV(self.kde1, {'bandwidth' : np.exp(np.linspace(log_b1 - 0.5 * np.log(dynamic_range), log_b1 + 0.5 * np.log(dynamic_range), gridsize))}, cv = cv, verbose = verbose, n_jobs = n_jobs)
            grid1.fit(X1)
            self.kde1 = grid1.best_estimator_
    def get_params(self, **kwargs: Any) -> Dict[str, Any]:
        return self.kde0.get_params(**kwargs)
    def set_params(self, **params: Dict[str, Any]) -> 'TwoClassKDE':
        self.kde0.set_params(**params)
        self.kde1.set_params(**params)
        return self
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the density model on the data. Returns vector of log-likelihood ratios of class 1 over class 0."""
        p1_est = (self.n1 + 1) / (self.n0 + self.n1 + 2)
        class_log_odds = np.log(p1_est) - np.log(1 - p1_est)
        scores0 = self.kde0.score_samples(X) if (self.n0 > 0) else np.zeros(len(X), dtype = float)
        scores1 = self.kde1.score_samples(X) if (self.n1 > 0) else np.zeros(len(X), dtype = float)
        return scores1 - scores0 + class_log_odds
    def score(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> float:
        """Compute the overall log-likelihood ratio under the model."""
        return self.score_samples(X).sum()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability estimates."""
        scores = self.score_samples(X)
        p0s = 1 / (1 + np.exp(scores))
        return np.array([p0s, 1 - p0s]).transpose()
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Log of probability estimates."""
        return np.log(self.predict_proba(X))
