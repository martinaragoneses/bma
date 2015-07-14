"""
Provides testing utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import linear_averaging

plt.style.use("ggplot")
plt.rcParams.update({'font.size': 16})



def simulate_data(nobs, weights, cor=0.5):
    """Simulate a simple dataset where 10 predictors are irrelevant and 5 increasingly relevant.
    Predictors are jointly gaussian with 0 mean, given correlation and variance 1.
    Residuals are gaussian with 0 mean and variance 1.

    Parameters
    ----------
    nobs : int {1, .., inf}
        number of samples to be drawn
    weights : np.ndarray
        model weights
    cor : float (-1, 1)
        correlation of the predictors

    Returns
    -------
    tup
        with [0] feature matrix and [1] response
    """

    ndim = len(weights)
    cov = (1 - cor) * np.identity(ndim) + cor * np.ones((ndim, ndim))
    mean = np.zeros(ndim)
    
    X = np.random.multivariate_normal(mean, cov, nobs)
    e = np.random.normal(0, 1, nobs)
    y = np.dot(X, weights) + e

    return (X, y)


def replicate_trial(trial, n):
    """Repeat a random trial n times.

    Parameters
    ----------
    trial : func
        call to the generating function
    n : int {1, .., inf}
        number of trials

    Returns
    -------
    np.ndarray
        where rows are trials and columns are variables
    """
    
    return np.array([trial() for i in range(n)])



# set coefficients
weights = np.hstack((np.zeros(10), np.arange(0.2, 1.2, 0.2)))

# simulate data
np.random.seed(2015)
X, y = simulate_data(100, weights, 0.9)

# full enumeration
enumerator = linear_averaging.LinearEnumerator(X, y, 15**2, 1/3)
enumerator.select()
enumerator.estimate()

# mcmc approximation
mc3 = linear_averaging.LinearMC3(X, y, 15**2, 1/3)
mc3.select(niter=10000, method="random")
mc3.estimate()
