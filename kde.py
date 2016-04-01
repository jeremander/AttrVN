import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV


class TwoClassKDE(object):
    """Class for Kernel Density Estimator on two labels. Likelihood ratio at a point is ratio of class-1 likelihood estimate to class-0 likelihood estimate, times the class odds, where this is calculated as the posterior mean estimate under Beta(1, 1) prior, given the observations. If no points are observed for one of the classes, a default (improper) uniform prior is assumed for that class. """
    def __init__(self, **kwargs):
        """Takes same parameters as KernelDensity estimator."""
        self.kde0 = KernelDensity(**kwargs)
        self.kde1 = KernelDensity(**kwargs)
    def fit(self, X, y):
        """Fits KDE models on the data. X is array of data points, y is array of 0-1 labels."""
        y = np.asarray(y, dtype = int)
        self.n0, self.n1 = (y == 0).sum(), (y == 1).sum()
        assert (self.n0 + self.n1 == len(y)), "y must be vector of 1's and 0's."
        X0, X1 = X[y == 0], X[y == 1]
        if (self.n0 > 0):
            self.kde0.fit(X0)
        if (self.n1 > 0):
            self.kde1.fit(X1)
    def fit_with_optimal_bandwidth(self, X, y, gridsize = 101, dynamic_range = 100, cv = 10, verbose = 0, n_jobs = 1):
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
    def get_params(self, **kwargs):
        return self.kde0.get_params(**kwargs)
    def set_params(self, **params):
        self.kde0.set_params(**params)
        self.kde1.set_params(**params)
        return self
    def score_samples(self, X):
        """Evaluate the density model on the data. Returns vector of log-likelihood ratios of class 1 over class 0."""
        p1_est = (self.n1 + 1) / (self.n0 + self.n1 + 2)
        class_log_odds = np.log(p1_est) - np.log(1 - p1_est)
        scores0 = self.kde0.score_samples(X) if (self.n0 > 0) else np.zeros(len(X), dtype = float)
        scores1 = self.kde1.score_samples(X) if (self.n1 > 0) else np.zeros(len(X), dtype = float)
        return scores1 - scores0 + class_log_odds
    def score(self, X, y = None):
        """Compute the overall log-likelihood ratio under the model."""
        return self.score_samples(X).sum()
    def predict_proba(self, X):
        """Probability estimates."""
        scores = self.score_samples(X)
        p0s = 1 / (1 + np.exp(scores))
        return np.array([p0s, 1 - p0s]).transpose()
    def predict_log_proba(self, X):
        """Log of probability estimates."""
        return np.log(self.predict_proba(X))


def demo(mu0 = np.array([2.0]), mu1 = np.array([0.0, 4.0]), sigma0 = np.array([1.41]), sigma1 = np.array([1.0, 1.0]), n0 = 100, n1 = 150, bandwidth = 0.5, gridsize = 101, dynamic_range = 100, cv = 10, verbose = 0, n_jobs = 1):
    """Compares optimal bandwidth strategy vs. fixed bandwidth for a multi-modal univariate distribution. User specifies means and standard deviations of GMMs for class 0 and 1, as well as the number of samples from each class."""
    plt.clf()
    X0 = np.array([sigma0[i] ** 2 * np.random.randn() + mu0[i] for i in np.random.randint(0, len(mu0), n0)]).reshape(n0, 1)
    X1 = np.array([sigma1[i] ** 2 * np.random.randn() + mu1[i] for i in np.random.randint(0, len(mu1), n1)]).reshape(n1, 1)
    plt.hist(X0, color = 'blue', alpha = 0.3, bins = 30)
    plt.hist(X1, color = 'red', alpha = 0.3, bins = 30)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0, dtype = int), np.ones(n1, dtype = int)])
    KDE = TwoClassKDE(bandwidth = bandwidth)
    KDE.fit(X, y)
    vals = np.linspace(X.min() - 1.0, X.max() + 1.0, 1000)
    scores = KDE.score_samples(vals.reshape(1000, 1))
    plt.plot(vals, scores, linewidth = 2, color = 'black')
    KDE.fit_with_optimal_bandwidth(X, y, gridsize, dynamic_range, cv, verbose, n_jobs)
    KDE.fit(X, y)
    scores2 = KDE.score_samples(vals.reshape(1000, 1))
    plt.plot(vals, scores2, linewidth = 2, linestyle = 'dashed', color = 'black')
    plt.show()



