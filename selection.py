"""
Provides routines for Bayesian model choice in a linear regression context, such that
    E[y|x_1, ..., x_n] = a + Xb
    where y is the response, b are coefficients and X are predictors.
and the full probability model is hierarchically specified as
    p[y, a, b, q, M] = p[y | a, b, q, M] * p[b | q, M] * p[a, q] * p[M]
    where M is the model/hypothesis and q is the residual precision

The routines compute the posterior distribution Pr[M|y]. This is accomplished either by computing the posterior probability of all possible models, or by sampling from the posterior distribution using MCMC methods.

The module is well suited for testing different proposal distributions within the MC3 framework.

Usage
-----
Analytical Estimation
    computes the exact posterior probability distribution over the model space. Exponential complexity in dimensions due to the growth of the model space.
    >>> LinearSelector(X, y)
Monte Carlo Estimation
    estimates the posterior probability distribution through a Metropolis-Hastings search. Much faster in high-dimensional model spaces, but less accurate.
    >>> LinearMC3(X, y)

References
----------
See Kass and Wassermann (1995) and Kass and Raftery (1995) for bayesian model choice and MC3
See Liang et al. (2008) for the specific probability model and its properties
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from collections import Counter
from scipy.special import binom
from diagnostics import MC2Diagnostics

plt.style.use("ggplot")



def log_sum_exp(sequence):
    """Compute the logarithm of a sum of exponentials in a stable way.
    If the difference between the smallest and the largest exponential is larger than the float-64 range, the routine will drop low-order terms.

    Parameters
    ----------
    sequence : array_like
        sequence of numbers to be exponentiated and added

    Returns
    -------
    float
        logarithm of the sum of exponentials of the sequence's elements
    """

    float_range = (-745, 705)
    lower = np.min(sequence)
    upper = np.max(sequence)

    if upper - lower < float_range[1] - float_range[0]:
        offset = (lower + upper) / 2
    else:
        print("Warning: Some very small terms have been dropped from a sum of exponentials to prevent overflow.")
        offset = upper - float_range[1]
        sequence = sequence[sequence - offset > float_range[0]]

    return offset + np.log(np.sum(np.e ** (sequence - offset)))


def log_gamma(x):
    """Compute the logarithm of the gamma function defined as log Gamma(x) = sum[from i = 1 to x - 1] log i

    Parameters
    ----------
    x : int {1, ..., inf}
        parameter of the gamma function

    Returns
    -------
    int
        value of the gamma function
    """

    return np.sum(np.log(np.arange(1, x)))


def binom_pmf(k, n, p):
    """Compute the density of the binomial distribution with given parameters.

    Parameters
    ----------
    k : int
        number of sucesses
    n : int {1, ..., inf}
        number of trials
    p : float [0, 1]
        probability of success of an individual trial

    Returns
    -------
    float
        density
    """
    
    return binom(n, k) * binom_prior_pmf(k, n, p)


def binom_prior_pmf(k, n, p):
    """Compute the prior probability of a model with k variables.

    Parameters
    ----------
    k : int
        number of variables in the model
    n : int {1, ..., inf}
        total number of predictors
    p : float [0, 1]
        prior inclusion probability of individual variables
    """
    
    if k < 0 or n < k:
        return 0
    return p ** k * (1 - p) ** (n - k)


class LinearSelector:
    """Provides an API for the analytical estimation of posterior model probabilities.

    This method computes posteriors for 2^d models,
    where d is the number of predictors. Use MC3 for larger d.

    Parameters
    ----------
    X : np.ndarray (nobs x ndim)
        prediction matrix
    y : np.ndarray (nobs x 1)
        response vector
    penalty_par : float (0, inf), default 1
        dimensionality penalty ("g")
    incl_par : float (0, 1), default 0.5
        prior inclusion probability ("p")
        the default implies a uniform prior over the model space

    Attributes
    ----------
    nobs : int
        number of observations
    ndim : int
        number of predictors
    X : np.ndarray
        prediction matrix
    y : np.ndarray
        response vector
    par : dict
        hyperparameters including "penalty" and "incl"
    posteriors: Counter
        dictionary of model posteriors where
        str(model) is the key and the posterior probability is the value
    """
    
    def __init__(self, X, y, penalty_par=1, incl_par=0.5):

        if X.shape[0] != len(y):
            raise InputError(
                (X.shape, len(y)),
                "The first dimension of X and y has to be aligned."
            )
        if float(penalty_par) <= 0:
            raise InputError(
                penalty_par,
                "penalty_par is restricted to (0, inf)."
            )
        if not 0 < float(incl_par) < 1:  
            raise InputError(
                incl_par,
                "incl_par is restricted to (0, 1)."
            )
        
        self.par = {
            "penalty": float(penalty_par),
            "incl": float(incl_par)
        }
        self.X = X
        self.y = y
        self.nobs, self.ndim = X.shape

        self._select()


    def test_single_coefficients(self, plot=False, errbars=None):
        """Evaluate the inclusion probability of single coefficients.
        
        Parameters
        ----------
        plot : bool, default False
            plot a bar chart with the inclusion probabilities
        errbars : array_like, default None
            error bars for the posterior estimates

        Returns
        -------
        np.ndarray
            (ndim x 1) vector of individual inclusion probabilities
        """

        weighted_models = np.array([
            posterior * np.fromstring(model[1:-1], sep=" ")
            for model, posterior in self.posteriors.items()
        ])

        if plot:
            plt.ylim(0, 1)
            plt.plot(
                range(1, self.ndim + 1),
                np.sum(weighted_models, 0),
                fmt="o",
                color="#444444",
                alpha=0.8
            )
            plt.show()
        
        return np.sum(weighted_models, 0)
        

    def test_joint_coefficients(self, indices):
        """Evaluate the joint inclusion probability of multiple coefficients.
        
        Parameters
        ----------
        indices : array_like
            indices of variables in X to be included in the test

        Returns
        -------
        float
            joint inclusion probability
        """
    
        return sum(
            posterior
            for model, posterior in self.posteriors.items()
            if np.all(np.fromstring(model[1:-1], sep=" ")[indices])
        )


    def model_size_distribution(self, plot=False):
        """Evaluate the posterior model size distribution.
        
        Parameters
        ----------
        plot : bool, default False
            plot a line chart of the prior (blue) and the posterior (red)

        Returns
        -------
        np.ndarray
            (ndim + 1 x 1) posterior model size probabilities
        """

        prior = [binom_pmf(i, self.ndim, self.par["incl"]) for i in range(self.ndim + 1)]
        posterior = np.zeros(self.ndim + 1)
        
        for model, post in self.posteriors.items():
            posterior[np.sum(np.fromstring(model[1:-1], sep=" "))] += post

        if plot:
            plt.plot(
                range(self.ndim + 1),
                prior,
                color="#66c2a5",
                ls="-",
                marker="o"
            )
            plt.plot(
                range(self.ndim + 1),
                posterior,
                color="#fc8d62",
                ls="-",
                marker="o"
            )
            plt.show()

        return posterior
        
    
    def _select(self):
        """Compute the posterior probability distribution
        by enumerating all (2^d) models.
        """
        
        models = np.array(list(product((0, 1), repeat = self.ndim)))
        
        # compute model probabilities
        priors = np.sum(models, 1) * np.log(self.par["incl"]) + (self.ndim - np.sum(models, 1)) * np.log(1 - self.par["incl"])
        likelihoods = np.array([
            self._likelihood(model)
            for model in models
        ])
        posteriors = np.e ** (priors + likelihoods - log_sum_exp(priors + likelihoods))
        
        # summarize
        self.posteriors = Counter({
            str(models[i]):posteriors[i]
            for i in range(len(models))
        })
        
        
    def _likelihood(self, model):
        """Compute the marginal likelihood of a given model.

        Parameters
        ----------
        model : array_like (ndim x 1)
            vector of 0 and 1 indicating whether to include a variable

        Returns
        -------
        float
           natural logarithm of the model's marginal likelihood
        """

        X = self.X[:,model == 1]
        ndim = X.shape[1]
        if len(X.shape) == 0:
            design = np.ones((self.nobs, 1))
        elif len(X.shape) == 1:
            X.shape = (self.nobs, 1)
            design = np.hstack((np.ones((self.nobs, 1)), X))
        else:
            design = np.hstack((np.ones((self.nobs, 1)), X))
        
        mle = np.linalg.solve(np.dot(design.T, design), np.dot(design.T, self.y))
        residuals = self.y - np.dot(design, mle)
        rsquared = 1 - np.var(residuals) / np.var(self.y)
        
        return (log_gamma((self.nobs - 1) / 2)
            - (self.nobs - 1) / 2 * np.log(np.pi)
            - 0.5 * np.log(self.nobs)
            - (self.nobs - 1) / 2 * np.log(np.dot(residuals, residuals))
            + (self.nobs - ndim - 1) / 2 * np.log(1 + self.par["penalty"])
            - (self.nobs - 1) / 2 * np.log(1 + self.par["penalty"] * (1 - rsquared)))


    
class LinearMC3(LinearSelector):
    """Provides an API for the monte carlo estimation of posterior model probabilities.
    
    Suitable for high-dimensional models.

    Parameters
    ----------
    X : np.ndarray (nobs x ndim)
        prediction matrix
    y : np.ndarray (nobs x 1)
        response vector
    penalty_par : float (0, inf), default 2.85^2
        dimensionality penalty ("g")
    incl_par : float (0, 1), default 0.5
        prior inclusion probability ("p")
        the default implies a uniform prior over the model space
    niter : int {1, .., inf}, default 10000
        number of draws from the distribution
    proposal : str {"random", "prior"}, default "random"
        strategy that determines MCMC proposal probabilities

    Attributes
    ----------
    nobs : int
        number of observations
    ndim : int
        number of predictors
    X : np.ndarray
        prediction matrix
    y : np.ndarray
        response vector
    par : dict
        parameters including "penalty", "incl", "niter", "proposal"
    posteriors: Counter
        dictionary of model posteriors where
        str(model) is the key and the posterior probability is the value
    diagnostics : dict
        dictionary of MCMC diagnostics per coefficient including
        "ess" (effective sample size)
        "ndiscard" (pre-equilibrium samples),
        "stderr" (standard error)
    """
    
    def __init__(self, X, y, penalty_par=1, incl_par=0.5, niter=10000, proposal="random"):

        if X.shape[0] != len(y):
            raise InputError(
                (X.shape, len(y)),
                "The first dimension of X and y has to be aligned."
            )
        if float(penalty_par) <= 0:
            raise InputError(
                penalty_par,
                "penalty_par is restricted to (0, inf)."
            )
        if not 0 < float(incl_par) < 1:  
            raise InputError(
                incl_par,
                "incl_par is restricted to (0, 1)."
            )
        if int(niter) < 1:  
            raise InputError(
                incl_par,
                "incl_par is restricted to {1, .., inf}."
            )
        if not proposal in ("random", "prior"):
            raise InputError(
                proposal,
                "proposal is restricted to {'random', 'prior'}."
            )
        
        self.par = {
            "penalty": float(penalty_par),
            "incl": float(incl_par),
            "niter": int(niter),
            "proposal": proposal == "random" and self._random_update or self._prior_update
        }
        self.X = X
        self.y = y
        self.nobs, self.ndim = X.shape

        self._select()


    def test_single_coefficients(self, plot=False):
        """Evaluate the inclusion probability of single coefficients.
        
        Parameters
        ----------
        plot : bool, default False
            plot chart with the inclusion probabilities

        Returns
        -------
        np.ndarray
            (ndim x 1) vector of individual inclusion probabilities
        """

        weighted_models = np.array([
            posterior * np.fromstring(model[1:-1], sep=" ")
            for model, posterior in self.posteriors.items()
        ])

        if plot:
            plt.ylim(0, 1)
            plt.errorbar(
                range(1, self.ndim + 1),
                np.sum(weighted_models, 0),
                fmt="o",
                color="#444444",
                alpha=0.5,
                yerr=1.96*self.diagnostics["stderr"],
                ecolor="#e41a1c"
            )
            plt.show()
        
        return np.sum(weighted_models, 0)

    
    def _select(self):
        """Estimate the posterior probability distribution through MCMC simulation.
        """

        draws = np.empty((self.par["niter"], self.ndim), dtype=bool)

        # pick initial model at random
        self.state = self.par["proposal"](np.random.binomial(1, self.par["incl"], self.ndim))
        
        # sample
        for i in range(self.par["niter"]):
            self._jump()
            draws[i,:] = self.state["model"]

        # ditch pre-equilibrium samples
        self.diagnostics = MC2Diagnostics(draws).summarize()
        self.draws = draws
        draws = draws[np.max(self.diagnostics["ndiscard"]):,:]

        # summarize
        counts = Counter()
        for i in range(draws.shape[0]):
            counts[str(np.array(draws[i,:], dtype=int)).replace("\n", "")] += 1
        self.posteriors = Counter({
            key:(counts[key] / sum(counts.values()))
            for key in counts
        })
    
        
    def _jump(self):
        """Attempt a state change in the markov chain.
        """

        # decide on an action
        add = np.random.binomial(1, self.state["growth"])

        # pick entering/leaving variable
        pick = np.random.choice(np.arange(self.ndim)[self.state["model"] != add])

        # profile new model
        model = np.copy(self.state["model"])
        model[pick] = not model[pick]
        candidate = self.par["proposal"](model)

        # proposal
        bayes_factor = candidate["lik"] - self.state["lik"]
        prior_odds = candidate["prior"] - self.state["prior"]
        posterior_odds = bayes_factor + prior_odds
        
        if np.sum(self.state["model"]) < np.sum(candidate["model"]):
            forward_proposal = self.state["growth"] / (self.ndim - np.sum(self.state["model"]))
            backward_proposal = (1 - candidate["growth"]) / np.sum(candidate["model"])
        else:
            forward_proposal = (1 - self.state["growth"]) / np.sum(self.state["model"])
            backward_proposal = candidate["growth"] / (self.ndim - np.sum(candidate["model"]))

        proposal_odds = np.log(backward_proposal) - np.log(forward_proposal)

        if posterior_odds + proposal_odds - np.log(np.random.uniform()) > 0:
            self.state = candidate


    def _random_update(self, model):
        """Compute the properties of the current state of the markov chain, using a random proposal rule.

        Parameters
        ----------
        model : array_like (ndim x 1)
            vector of 0 and 1 indicating whether to include a variable

        Returns
        -------
        dict
            includes the needed information to evaluate state changes
            "model" (current variable indicators)
            "growth" (the growth probability)
            "prior" (the model's prior probability)
            "lik" (the model's marginal likelihood)
        """
        
        return {
            "model": np.array(model, dtype=bool),
            "growth": 1 - np.sum(model) / self.ndim,
            "prior": binom_prior_pmf(np.sum(model), self.ndim, self.par["incl"]),
            "lik": self._likelihood(model)
        }


    def _prior_update(self, model):

        prob_dplus = binom_pmf(
            np.sum(model) + 1,
            self.ndim,
            self.par["incl"]
        )
        prob_dminus = binom_pmf(
            np.sum(model) - 1,
            self.ndim,
            self.par["incl"]
        )
        return {
            "model": np.array(model, dtype=bool),
            "growth": prob_dplus / (prob_dplus + prob_dminus),
            "prior": binom_prior_pmf(np.sum(model), self.ndim, self.par["incl"]),
            "lik": self._likelihood(model)
        }
    

    def _condn_update(self, model, magicn=22):
        """Compute the properties of the current state of the markov chain, using a condition-number proposal rule.

        Parameters
        ----------
        model : array_like (ndim x 1)
            vector of 0 and 1 indicating whether to include a variable

        Returns
        -------
        dict
            includes the needed information to evaluate state changes
            "model" (current variable indicators)
            "growth" (the growth probability)
            "prior" (the model's prior probability)
            "lik" (the model's marginal likelihood)
        """

        X = self.X[:,model == 1]
        if np.sum(model) < 2:
            cond_nr = 1
        else:
            cond_nr = np.linalg.cond(np.dot(X.T, X))
        return {
            "model": np.array(model, dtype=bool),
            "growth": (2 / (1 + np.e ** (cond_nr / magicn))) ** (np.all(model) and np.inf or 1) ** (not np.all(np.logical_not(model))),
            "prior": binom_prior_pmf(np.sum(model), self.ndim, self.par["incl"]),
            "lik": self._likelihood(model)
        }



class InputError(Exception):

    def __init__(self, expr, msg):
        self.expr = expr
        self.msg = msg
        
    def __str__(self):
        return str(self.expr) + ": " + str(self.msg)
