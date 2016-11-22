from .vngraph import *
import numpy as np
import scipy.stats
from scipy.optimize import minimize
from scipy.special import expit

np.seterr(over = 'ignore')

def log_one_plus_exp(x):
    """The function log(1 + exp(x))."""
    if (x < -50):
        return 0.0
    elif (x > 50):
        return x
    else:
        return np.log(1 + np.exp(x))      


class GMM():
    """Class representing a multivariate Gaussian mixture model."""
    def __init__(self, means, weights = None, covs = None):
        """means is K x m matrix of means, weights is K-long vector of mixture probabilities, and covs is either a K-long list of m x m covariance matrices, or a K-long list of diagonal matrix coefficients."""
        self.K, self.m = means.shape
        self.means = means 
        self.weights = (np.ones(self.K) / self.K) if (weights is None) else (weights / sum(weights))
        assert (len(self.weights) == self.K)
        covs = 1.0 if (covs is None) else covs
        if isinstance(covs, (int, float)):
            self.covs = np.array([covs * np.eye(self.m) for i in range(self.K)])
        elif (len(covs.shape) == 1):
            self.covs = np.array([c * np.eye(self.m) for c in covs])
        else:
            self.covs = covs
        self.cumsums = np.cumsum(self.weights)
        self.mvns = [scipy.stats.multivariate_normal(mu, cov) for (mu, cov) in zip(self.means, self.covs)]
    def sample(self, n):
        X = np.zeros((n, self.m), dtype = float)
        for i in range(n):
            r = np.random.rand()
            for j in range(self.K):
                if (r < self.cumsums[j]):
                    X[i] = self.mvns[j].rvs()
                    break
        return X

class GMMPrior():
    """Class generating a random GMM in m-dimensional Euclidean space."""
    def __init__(self, m, K, mu = None, cov = None, alpha = None):
        """Initialize from m (dimension), K (number of centers), mu (mean vector for the centers), cov (covariance matrix for the centers), alpha (K-long Dirichlet distribution vector for mixture probabilities)."""
        self.m = m 
        self.K = K
        if (mu is None):
            self.mu = np.zeros(m)  # default zero
        else:
            assert (len(self.mu) == m)
            self.mu = mu
        if (cov is None):
            self.cov = np.eye(m)  # default identity
        else:
            assert (cov.shape == (K, K))
            self.cov = cov
        if (alpha is None):
            self.alpha = 1000 * np.ones(K)  # nearly uniform default
        else:
            assert (len(alpha) == K)
            self.alpha = alpha
        self.mvn = scipy.stats.multivariate_normal(self.mu, self.cov)
        self.dirichlet = scipy.stats.dirichlet(self.alpha)
    def sample(self, covs = None):
        """Samples a GMM from the prior distribution on GMM parameters. covs is a list of covariance matrices, a list of coefficients for diagonal covariance matrices, or a constant to be used for all diagonal covariance matrices (default 0.1)."""
        means = self.mvn.rvs(self.K)
        covs = 0.1 if (covs is None) else covs
        weights = self.dirichlet.rvs()[0]
        return GMM(means, weights, covs)


class LatentFactorModel():
    """Class representing a latent factor model for n items in R^m."""
    def __init__(self, X):
        """Initialize from an n x m matrix."""
        self.X = X
        self.n, self.m = self.X.shape

class LFRG(LatentFactorModel):
    def __init__(self, X, s, link = 'logistic'):
        """Inputs are n x m factor matrix X, n-long vector of "sociability" coefficients, and a link function specification (default 'logistic')."""
        super().__init__(X)
        assert (len(s) == self.n)
        self.s = s
        if (link == 'logistic'):
            self.ginv = expit
        else:
            raise ValueError("Invalid link type '%s'." % str(link))
    def sample(self):
        """Samples a sparse n x n graph via P(u ~ v) = ginv(X_u X_v^T + s_u + s_j)."""
        g = UndirectedGraph()
        g.add_nodes_from(range(self.n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                p = self.ginv(np.dot(self.X[i], self.X[j]) + self.s[i] + self.s[j])
                if (np.random.rand() <= p):
                    g.add_edge(i, j)
        return g

gmm = GMM(np.array([[-3.0, 3.0], [3.0, 3.0], [3.0, -3.0], [-3.0, -3.0]]))


class LFRG_MLE():
    def __init__(self, G, m):
        self.G = G
        self.m = m
        self.n = self.G.number_of_nodes()
        m, n = self.m, self.n
        self.zero_indices = [j for i in range(m) for j in range(i * m + i + 1, (i + 1) * m)]
        self.zero_indices_set = set(self.zero_indices)
        self.num_constrained = len(self.zero_indices)
        self.x0 = np.random.normal(size = n * m + n - self.num_constrained)  # default init
    def split(self, x):
        m, n = self.m, self.n
        X = np.zeros((n, m), dtype = float)
        x_ind = 0
        for ctr in range(n * m):
            if (ctr not in self.zero_indices_set):
                X[ctr // m, ctr % m] = x[x_ind]
                x_ind += 1
        return (X, x[(n * m) - self.num_constrained:])
    def join(self, X, s):
        m, n = self.m, self.n
        x = np.zeros(n * m + n - self.num_constrained, dtype = float)
        x_ind = 0
        for ctr in range(n * m):
            if (ctr not in self.zero_indices_set):
                x[x_ind] = X[ctr // m, ctr % m]
                x_ind += 1
        x[x_ind:] = s
        return x
    def neg_log_likelihood_and_gradient(self, x):
        """Returns exact log-likelihood and gradient (computation is not sparse)."""
        m, n = self.m, self.n
        if (not hasattr(self, 'A')):
            self.ones = np.ones((n, 1), dtype = float)
            self.A = np.asarray(nx.adjacency_matrix(self.G).todense(), dtype = float)
            self.B = np.ones((n, n), dtype = float) - np.eye(n)
        (X, s) = self.split(x)
        M = np.dot(X, X.T) + np.dot(self.ones, s.reshape((1, n))) + np.dot(s.reshape((n, 1)), self.ones.T)
        M2 = np.log1p(np.exp(M))
        flags = np.isinf(M2)
        M2[flags] = 0
        M2 = M2 + flags * M
        log_likelihood = (0.5 * np.dot(self.ones.T, np.dot(self.A * M, self.ones)) - 0.5 * np.dot(self.ones.T, np.dot(self.B * M2, self.ones)))[0, 0]
        M2 = expit(M)
        BM2 = self.B * M2
        X_grad = np.dot(self.A, X) - np.dot(BM2, X)
        s_grad = np.dot(self.A, self.ones) - np.dot(BM2, self.ones)
        log_likelihood_gradient = self.join(X_grad, s_grad.reshape(n))  
        return (-log_likelihood, -log_likelihood_gradient)   
    def approx_neg_log_likelihood_and_gradient(self, x):
        m, n = self.m, self.n
        if (not hasattr(self, 'edge_indices')):
            self.edge_dict = {edge : k for (k, edge) in enumerate(self.G.edges_iter())}
            self.nonneighbor_counts = n - 1 - self.G.degrees()
        (X, s) = self.split(x)
        s_bar = np.mean(s)
        log_likelihood = 0.0
        X_grad = np.zeros((n, m), dtype = float)
        s_grad = np.zeros(n, dtype = float)
        for i in range(n):
            b = s[i] + s_bar
            #log_likelihood -= self.nonneighbor_counts[i] * s[i]
            log_likelihood -= self.nonneighbor_counts[i] * log_one_plus_exp(b)
            #s_grad[i] -= self.nonneighbor_counts[i]
            s_grad[i] -= self.nonneighbor_counts[i] * expit(b)
            for j in self.G.neighbors_iter(i):
                if (i < j):
                    c = np.dot(X[i], X[j]) + s[i] + s[j]
                    log_likelihood += (c - log_one_plus_exp(c))
                    d = 1 / (1 + np.exp(c))
                    X_grad[i] += X[j] * d
                    X_grad[j] += X[i] * d
                    s_grad[i] += d
                    s_grad[j] += d
        log_likelihood_gradient = self.join(X_grad, s_grad.reshape(n))  
        return (-log_likelihood, -log_likelihood_gradient)
    def MLE_exact(self, x0 = None, method = 'L-BFGS-B', maxiter = 100, tol = None, verbose = False):
        """Optimizes parameters by maximizing exact log-likelihood (computation is not sparse)."""
        x0 = self.x0 if (x0 is None) else x0  
        self.x0 = minimize(self.neg_log_likelihood_and_gradient, x0, jac = True, method = method, tol = tol, options = {'maxiter' : maxiter, 'disp' : verbose})['x']
        return self.split(self.x0)
    def MLE_approx(self, x0 = None, method = 'L-BFGS-B', maxiter = 100, tol = None, verbose = False):
        """Optimizes parameters by maximizing approx log-likelihood (computation is sparse)."""
        x0 = self.x0 if (x0 is None) else x0  
        self.x0 = minimize(self.approx_neg_log_likelihood_and_gradient, x0, jac = True, method = method, tol = tol, options = {'maxiter' : maxiter, 'disp' : verbose})['x']
        return self.split(self.x0)
