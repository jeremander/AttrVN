#!/usr/bin/env python3
"""Compares optimal bandwidth strategy vs. fixed bandwidth for a multi-modal univariate distribution. User specifies means and standard deviations of GMMs for class 0 and 1, as well as the number of samples from each class."""

import matplotlib.pyplot as plt
import numpy as np

from rags.kde import TwoClassKDE

mu0 = np.array([2.0])
mu1 = np.array([0.0, 4.0])
sigma0 = np.array([1.41])
sigma1 = np.array([1.0, 1.0])
n0 = 100
n1 = 150
bandwidth = 0.5
gridsize = 101
dynamic_range = 100
cv = 10
verbose = 0
n_jobs = 1

def demo():
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

if __name__ == '__main__':
    demo()