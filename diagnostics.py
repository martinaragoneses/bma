"""
Provides MCMC diagnostics to evaluate convergence and sample dependence.

Sample dependence is controlled by the integrated autocorrelation time
    IAT = (0.5 + sum[t = 1][inf](acf(t))
    where acf(t) is the autocorrelation function at lag t.
Deviation from the equilibrium distribution is controlled by the exponential autocorrelation time
    EAT = sup[t](- t / log(|acf(t)|))

Usage
-----
Initialize by constucting
    >>> MC2Diagnostics(X)

References
----------
See Sokal (1997) for justification and estimation of IAT and EAT.
"""

import numpy as np
import pdb
from collections import Counter



def get_kendalltau_bin(sample1, sample2):
    """Calculate Kendall's tau-b in O(n) time.
    
    Parameters
    ----------
    sample1/2 : array_like
        binary sequences

    Returns
    -------
    float
        Kendall's tau-b or 0 if undefined
    """

    c = Counter()
    for i in range(len(sample1)):
        c[(sample1[i], sample2[i])] += 1

    conc = c[(1, 1)] * c[(0, 0)]
    nconc = c[(1, 0)] * c[(0, 1)]
    ties1 = c[(0, 0)] * c[(0, 1)] + c[(1, 1)] * c[(1, 0)]
    ties2 = c[(0, 0)] * c[(1, 0)] + c[(1, 1)] * c[(0, 1)]

    try:
        return (conc - nconc) / ((conc + nconc + ties1) * (conc + nconc + ties2)) ** 0.5
    except ZeroDivisionError:
        return 0


def est_bin_acf(series):
    """Estimate the binary autocorrelation function.

    Parameters
    ----------
    series : array_like
        time series
    
    Returns
    -------
    np.ndarray
        acf up to lag n // 100, starting at lag 1
    """

    return [
        get_kendalltau_bin(series[lag:], series[:-lag])
        for lag in range(1, 100 + 1)
    ]


def est_int_autocor(acf, tradeoff_par=6):
    """Estimate the integrated autocorrelation time.
    Since there is a bias-variance tradeoff involved in the estimation, the acf is integrated up to lag l such that l is the smallest int for which
        l >= tradeoff_par * (0.5 + sum[t = 1][l](acf(t))

    Parameters
    ----------
    acf : array_like
        autocorrelation function starting at lag 1
    tradeoff_par : int {1, .., inf}, default 6
        governs the bias-variance tradeoff in the estimation. A higher parameter lowers the bias and increases the variance

    Returns
    -------
    float
        estimate of the acf's integrated autocorrelation time
    """
    
    int_autocor = 0.5
    for i in range(len(acf)):
        int_autocor += acf[i]
        if i + 1 >= tradeoff_par * int_autocor:
            return int_autocor
    return int_autocor



def est_exp_autocor(acf):
    """Estimate the exponential autocorrelation time.
    This method is very sensitive to post-decay noise in the acf.

    Parameters
    ----------
    acf : array_like
        autocorrelation function starting at lag 1

    Returns
    -------
    float
        estimate of the acf's exponential autocorrelation time
    """
    
    return np.nanmax([
        (lag + 1) / -np.log(np.abs(acf[lag]))
        for lag in range(len(acf))
     ])



class MC2Diagnostics:
    """Provides the API for the computation of MCMC diagnostics.

    Parameters
    ----------
    samples : np.ndarray (nsamples x nseries)
        matrix where each row (f1(x), ..., fn(x)) is a vector of functions of the current MC state

    Attributes
    ----------
    samples : np.ndarray (nsamples x nseries)
        matrix containing sample rows of the statistics of interest
    int_autocor: np.ndarray (nseries)
        estimated integrated autocorrelation time for the statistics of interest
    """
    
    def __init__(self, samples):
        
        self.samples = samples
        self.acfs = [est_bin_acf(series) for series in self.samples.T]
        self.int_autocor = np.array([est_int_autocor(acf) for acf in self.acfs])

        
    def summarize(self):
        """Summarize sampling properties of all series.

        Returns
        -------
        dict
            summary statistics for all series including
            "ess" (effective sample size regarding mean estimation)
            "ndiscard" (estimated number of pre-equilibrium samples)
            "stderr" (standard error of the mean estimator)
        """
        
        ndiscard = 20 * self.int_autocor
        ess = (self.samples.shape[0] - ndiscard) / 2 / self.int_autocor
        means = np.array([np.mean(series) for series in self.samples.T])
        stderr = (np.array([np.var(series) for series in self.samples.T]) / ess) ** 0.5
        
        return {
            "ess": ess,
            "ndiscard": ndiscard,
            "stderr": stderr
        }
