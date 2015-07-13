"""
Provides routines for Bayesian model averaging. You can adapt these to your model by providing its marginal likelihood function and its prior probability function.

The routines compute the posterior distribution Pr[M|y]. This is accomplished either by computing the posterior probability of all possible models, or by sampling from the posterior distribution using MCMC methods.

References
----------
See Kass and Wassermann (1995) and Kass and Raftery (1995) for bayesian model averaging and MC3
"""

import numpy as np
from itertools import product
from collections import Counter
from scipy.special import binom
from diagnostics import MC2Diagnostics



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



class Enumerator(object):
    """Computes the posterior probability distribution over a model space defined by a marginal likelihood and a prior.

    This method computes 2^d posteriors, where d is the number of predictors. Use MC3 for larger d.

    Parameters
    ----------
    X : np.ndarray (nobs x ndim)
        predictor matrix
    y : np.ndarray (nobs x 1)
        response vector
    likelihood_func : func
        function that returns the log marginal likelihood of a given model
    prior_func : func
        function that returns the prior probability of a given model
    
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
    posteriors: Counter
        posterior distribution over the model space where
        str(model) is the key and the posterior probability is the value
    """
    
    def __init__(self, X, y, likelihood_func, prior_func):
        
        self.X = X
        self.y = y
        self.nobs, self.ndim = X.shape

        self._select()


    def test_single_coefficients(self):
        """Evaluate the inclusion probability of single coefficients.
        
        Returns
        -------
        np.ndarray
            (ndim x 1) vector of individual inclusion probabilities
        """

        weighted_models = np.array([
            posterior * np.fromstring(model[1:-1], dtype=bool, sep=" ")
            for model, posterior in self.posteriors.items()
        ])
        
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


    def model_size_distribution(self):
        """Evaluate the posterior model size distribution.
        
        Returns
        -------
        np.ndarray
            (ndim + 1 x 1) posterior model size probabilities
        """
        
        posterior = np.zeros(self.ndim + 1)
        
        for model, post in self.posteriors.items():
            posterior[np.sum(np.fromstring(model[1:-1], sep=" "))] += post

        return posterior
        
    
    def _select(self):
        """Compute the posterior probability distribution
        by enumerating all (2^d) models.
        """
        
        models = np.array(list(product((0, 1), repeat = self.ndim)))
        
        # compute model probabilities
        priors = np.array([
            np.log(self.prior_func(np.sum(model), self.ndim))
            for model in models
        ])
        likelihoods = np.array([
            self.likelihood_func(self.X[:, model == 1], self.y)
            for model in models
        ])
        posteriors = np.e ** (priors + likelihoods - log_sum_exp(priors + likelihoods))
        
        # summarize
        self.posteriors = Counter({
            str(models[i]):posteriors[i]
            for i in range(len(models))
        })
        
        

class MC3(Enumerator):
    """Computes the posterior probability distribution over a model space defined by a marginal likelihood and a prior.
    
    Suitable for high-dimensional models.

    Parameters
    ----------
    X : np.ndarray (nobs x ndim)
        predictor matrix
    y : np.ndarray (nobs x 1)
        response vector
    likelihood_func : func
        function that returns the log marginal likelihood of a given model
    prior_func : func
        function that returns the prior probability of a given model
    niter : int {1, .., inf}
        number of draws from the distribution
    proposal : str {"random", "prior"}
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
    posteriors: Counter
        posterior distribution over the model space where
        str(model) is the key and the posterior probability is the value
    par : dict
        MCMC parameters including "niter", "proposal"
    diagnostics : dict
        MCMC diagnostics per coefficient including
        "ess" (effective sample size)
        "ndiscard" (pre-equilibrium samples),
        "stderr" (standard error)
    """
    
    def __init__(self, X, y, likelihood_func, prior_func, niter, proposal):
        
        self.par = {
            "niter": int(niter),
            "proposal": proposal == "random" and self._random_update or self._prior_update
        }
        self.X = X
        self.y = y
        self.nobs, self.ndim = X.shape

        self._select()
        
    
    def _select(self):
        """Estimate the posterior probability distribution through MCMC simulation.
        """

        draws = np.empty((self.par["niter"], self.ndim), dtype=bool)

        # pick initial model at random
        self.state = self.par["proposal"](np.zeros(self.ndim))
        
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
            vector of 0 and 1 indicating whether a variable is included

        Returns
        -------
        dict
            includes the needed information to evaluate state changes
            "model" (current variable indicators)
            "growth" (the growth probability)
            "prior" (the model's log prior probability)
            "lik" (the model's log marginal likelihood)
        """
        
        return {
            "model": np.array(model, dtype=bool),
            "growth": 1 - np.sum(model) / self.ndim,
            "prior": np.log(self.prior_func(np.sum(model), self.ndim)),
            "lik": self.likelihood_func(self.X[:, model == 1], self.y)
        }


    def _prior_update(self, model):
        """Compute the properties of the current state of the markov chain.

        Parameters
        ----------
        model : array_like (ndim x 1)
            vector of 0 and 1 indicating whether a variable is included

        Returns
        -------
        dict
            includes the needed information to evaluate state changes
            "model" (current variable indicators)
            "growth" (the growth probability)
            "prior" (the model's log prior probability)
            "lik" (the model's log marginal likelihood)
        """
        
        prob_dplus = binom(self.ndim, np.sum(model) + 1) * self.prior_func(
            np.sum(model) + 1,
            self.ndim
        )
        prob_dminus = binom(self.ndim, np.sum(model) - 1) * self.prior_func(
            np.sum(model) - 1,
            self.ndim
        )
        return {
            "model": np.array(model, dtype=bool),
            "growth": prob_dplus / (prob_dplus + prob_dminus),
            "prior": np.log(self.prior_func(np.sum(model), self.ndim)),
            "lik": self.likelihood_func(self.X[:, model == 1], self.y)
        }
    

    def _condn_update(self, model, magicn=22):
        """Compute the properties of the current state of the markov chain, using a condition-number proposal rule.

        Parameters
        ----------
        model : array_like (ndim x 1)
            vector of 0 and 1 indicating whether a variable is included

        Returns
        -------
        dict
            includes the needed information to evaluate state changes
            "model" (current variable indicators)
            "growth" (the growth probability)
            "prior" (the model's log prior probability)
            "lik" (the model's log marginal likelihood)
        """

        X = self.X[:,model == 1]
        if np.sum(model) < 2:
            cond_nr = 1
        else:
            cond_nr = np.linalg.cond(np.dot(X.T, X))
        return {
            "model": np.array(model, dtype=bool),
            "growth": (2 / (1 + np.e ** (cond_nr / magicn))) ** (np.all(model) and np.inf or 1) ** (not np.all(np.logical_not(model))),
            "prior": np.log(self.prior_func(np.sum(model), self.ndim)),
            "lik": self.likelihood_func(self.X[:, model == 1], self.y)
        }



class InputError(Exception):

    def __init__(self, expr, msg):
        self.expr = expr
        self.msg = msg
        
    def __str__(self):
        return str(self.expr) + ": " + str(self.msg)
