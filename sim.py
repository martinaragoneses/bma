"""
Contains utilities that implement the simulation design.
"""

import numpy as np
import matplotlib.pyplot as plt
import linear_averaging
from scipy.special import binom

plt.style.use("ggplot")
plt.rcParams.update({'font.size': 16})



def simulate_data_d15(nobs):
    """Simulate a simple dataset where 10 predictors are irrelevant and 5 increasingly relevant.
    Predictors are jointly gaussian with 0 mean, covariance 0.5 and variance 1.
    Residuals are gaussian with 0 mean and variance 1.

    Parameters
    ----------
    nobs : int {1, .., inf}
        number of samples to be drawn

    Returns
    -------
    tup
        with [0] feature matrix and [1] response
    """
    
    cov = 0.5 * (np.identity(15) + np.ones((15, 15)))
    mean = np.zeros(15)
    weights = np.hstack((np.zeros(10), np.arange(0.2, 1.2, 0.2)))
    
    X = np.random.multivariate_normal(mean, cov, nobs)
    e = np.random.normal(0, 1, nobs)
    y = np.dot(X, weights) + e

    return (X, y)


def simulate_data_d30(nobs):
    """Simulate a simple dataset where 20 predictors are irrelevant and 10 increasingly relevant.
    Predictors are jointly gaussian with 0 mean, covariance 0.5 and variance 1.
    Residuals are gaussian with 0 mean and variance 1.

    Parameters
    ----------
    nobs : int {1, .., inf}
        number of samples to be drawn

    Returns
    -------
    tup
        with [0] feature matrix and [1] response
    """
    
    cov = 0.5 * (np.identity(30) + np.ones((30, 30)))
    mean = np.zeros(30)
    weights = np.hstack((np.zeros(20), np.arange(0.1, 1.1, 0.1)))
    
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



np.random.seed(2015)
X, y = simulate_data_d15(100)
model1 = linear_averaging.LinearMC3(X, y, 15**2, 1/3)
model1.select(10000, "random")
model2 = linear_averaging.LinearEnumerator(X, y, 15**2, 1/3)
model2.select()
