"""
Provides routines for Markov Chain Monte Carlo (MCMC) sampling.

References
----------
See Hastings (1970) for details on the Metropolis-Hastings algorithm.
See Sokal (1997) for MCMC diagnostics.
"""

import numpy as np
from collections import Counter



def get_kendalltau_bin(sample1, sample2):
    """Calculate Kendall's tau-b in O(n) time.

    Parameters
    ----------
    sample1/2 : array_like in {0, 1}^ndim
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
    series : array_like in {0, 1}^d
        binary time series

    Returns
    -------
    np.ndarray
        acf up to lag n // 100, starting at lag 1
    """

    return [
        get_kendalltau_bin(series[lag:], series[:-lag])
        for lag in range(1, 100 + 1)
    ]


def est_acf(series):
    """Estimate the autocorrelation function.

    Parameters
    ----------
    series : array_like in R^d
        binary time series

    Returns
    -------
    np.ndarray
        acf up to lag n // 100, starting at lag 1
    """

    mean = np.mean(series)
    var = np.mean(series ** 2) - mean ** 2

    acf =  np.array([
        (np.mean(series[lag:] * series[:-lag]) - mean ** 2) / var
        for lag in range(1, 100 + 1)
    ])

    acf[np.isnan(acf)] = 0
    return acf


def est_int_autocor(acf, tradeoff_par=6):
    """Estimate the integrated autocorrelation time.
    Since there is a bias-variance tradeoff involved in the estimation, the acf is integrated up to lag l such that l is the smallest int for which
        l >= tradeoff_par * (0.5 + sum[t = 1][l](acf(t))

    Parameters
    ----------
    acf : array_like in (0, 1)^ndim
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
    acf : array_like in (0, 1)^ndim
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



class MetropolisSampler(object):
    """Generic implementation of the Metropolis-Hastings algorithm.

    Draws dependent samples from a probability distribution.

    Parameters
    ----------
    rv_prob_func : func
        log probability measure on the random variable
    proposal_func : func
        draw from the proposal distribution given the current state
    proposal_prob_func : func
        log probability measure on the proposal distribution

    Attributes
    ----------
    draws : np.ndarray
        array of draws from the random variable
    diagnostics : dict
        summary statistics for all dimensions including
        "ess" (effective sample size)
        "ndiscard" (estimated number of pre-equilibrium samples)
        "stderr" (standard error of the mean estimator)
    """

    def __init__(self, rv_prob_func, proposal_func, proposal_prob_func):

        self._get_rv_prob = rv_prob_func
        self._propose = proposal_func
        self._get_proposal_prob = proposal_prob_func


    def _run(self, init, niter=100000):
        """Run the sampler.

        Parameters
        ----------
        init : float or array_like in R^ndim
            initial state of the Markov chain
        niter : int {1, .., inf}
            number of iterations
        """

        self.draws = np.empty((niter, len(init)), dtype=np.array(init).dtype)
        state = {"draw": init, "prob": self._get_rv_prob(init)}

        for i in range(niter):
            candidate = {"draw": self._propose(state["draw"])}
            candidate["prob"] = self._get_rv_prob(candidate["draw"])
            if self._decide(state, candidate):
                state = candidate
            self.draws[i, :] = state["draw"]

        # discard burn-in
        self._diagnose()
        self.draws = self.draws[np.max(self.diagnostics["ndiscard"]):,:]


    def _decide(self, state, proposal):
        """Apply the Metropolis-Hastings decision rule to a candidate state.

        Parameters
        ----------
        proposal : np.ndarray in R^ndim
            candidate MC state
        state : np.ndarray in R^ndim
            current MC state

        Returns
        -------
        bool
            decision
        """

        odds_ratio = proposal["prob"] - state["prob"]
        proposal_ratio = self._get_proposal_ratio(state, proposal)

        if odds_ratio + proposal_ratio > np.log(np.random.uniform()):
            return True
        else:
            return False


    def _get_proposal_ratio(self, state, candidate):
        """Compute the ratio of proposal probabilities.

        Parameters
        ----------
        proposal : np.ndarray in R^ndim
            candidate MC state
        state : np.ndarray in R^ndim
            current MC state

        Returns
        -------
        float
            log ratio
        """

        forward_prob = self._get_proposal_prob(
            candidate["draw"],
            state["draw"]
        )
        backward_prob = self._get_proposal_prob(
            state["draw"],
            candidate["draw"]
        )

        return backward_prob - forward_prob


    def _diagnose(self):
        """Compute performance statistics.
        """

        acfs = [est_acf(series) for series in self.draws.T]
        int_autocor = np.array([est_int_autocor(acf) for acf in acfs])
        ndiscard = 20 * int_autocor
        ess = (self.draws.shape[0] - ndiscard) / 2 / int_autocor
        means = np.array([np.mean(series) for series in self.draws.T])
        stderr = (np.array([np.var(series) for series in self.draws.T]) / ess) ** 0.5

        self.diagnostics = {
            "ess": ess,
            "ndiscard": ndiscard,
            "stderr": stderr
        }
