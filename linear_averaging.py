"""
Provides routines for Bayesian model choice in a linear regression context, such that
    E[y|x_1, ..., x_n] = a + Xb
    where y is the response, b are coefficients and X are predictors.

The full probability model is given by
    p[y, a, b, q, M] = p[y | a, b, q, M] * p[b | q, M] * p[a, q] * p[M]
    where M is the model/hypothesis and q is the residual precision

References
----------
See Liang et al. (2008) for the specific probability model and its properties
"""

import numpy as np
import core
import linear_regression



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
    

def binom_prior_pmf(k, n, p):
    """Compute the prior probability of a model with k variables.

    Parameters
    ----------
    k : int {1, ..., n}
        number of variables in the model
    n : int {1, ..., inf}
        total number of predictors
    p : float [0, 1]
        prior inclusion probability of individual variables

    Returns
    -------
    float
        prior probability
    """
    
    if k < 0 or n < k:
        return 0
    return p ** k * (1 - p) ** (n - k)


def marginal_likelihood(X, y, g):
    """Compute the marginal likelihood of the linear model with a g-prior on betas.

    Parameters
    ----------
    X : np.ndarray (nobs x ndim)
        regressor matrix
    y : np.ndarray
        response vector
    g : dimensionality penalty

    Returns
    -------
    float
        natural logarithm of the model's marginal likelihood
    """

    nobs, ndim = X.shape
    
    if len(X.shape) == 0:
        design = np.ones((nobs, 1))
    elif len(X.shape) == 1:
        X.shape = (nobs, 1)
        design = np.hstack((np.ones((nobs, 1)), X))
    else:
        design = np.hstack((np.ones((nobs, 1)), X))
    
    mle = np.linalg.solve(np.dot(design.T, design), np.dot(design.T, y))
    residuals = y - np.dot(design, mle)
    rsquared = 1 - np.var(residuals) / np.var(y)
    
    return (log_gamma((nobs - 1) / 2)
        - (nobs - 1) / 2 * np.log(np.pi)
        - 0.5 * np.log(nobs)
        - (nobs - 1) / 2 * np.log(np.dot(residuals, residuals))
        + (nobs - ndim - 1) / 2 * np.log(1 + g)
        - (nobs - 1) / 2 * np.log(1 + g * (1 - rsquared)))



class LinearEnumerator(core.Enumerator):
    """Computes the posterior probability distribution over the space of linear regression models.

    This method computes 2^d posteriors, where d is the number of predictors. Use MC3 for larger d.
    
    Parameters
    ----------
    X : np.ndarray (nobs x ndim)
        prediction matrix
    y : np.ndarray (nobs x 1)
        response vector
    penalty_par : float (0, inf)
        dimensionality penalty ("g")
    incl_par : float (0, 1)
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
    posteriors: Counter
        posterior distribution over the model space where
        str(model) is the key and the posterior probability is the value
    estimates: dict
        point estimates of the model parameters including
        "coefficients" and "res_precision"
    """

    def __init__(self, X, y, penalty_par, incl_par):

        self.X = X
        self.y = y
        self.nobs, self.ndim = X.shape
        self.likelihood_func = lambda X, y: marginal_likelihood(X, y, penalty_par)
        self.prior_func = lambda k, n: binom_prior_pmf(k, n, incl_par)

        self._select()
        self._estimate(penalty_par)


    def predict(self, X_new):
        """Give point predictions for a new set of observations.

        Parameters
        ----------
        X_new : np.ndarray
            prediction matrix

        Returns
        -------
        np.ndarray
            vector of point estimates
        """

        if X_new.shape[0] == self.X.shape[1]:
            design = np.hstack((np.ones((1, 1)), X_new[np.newaxis]))
        else:
            design = np.hstack((np.ones((X_new.shape[0], 1)), X_new))
            
        return np.dot(design, self.estimates["coefficients"])


    def residual_dist(self, ndraws=1000):
        """Draw from the posterior distribution of the variance of residuals.

        Parameters
        ----------
        ndraws : int, default 1000
            number of draws

        Retrurns
        --------
        np.ndarray
            vector of draws
        """

        g = max(self.X.shape[0], self.X.shape[1] ** 2)
        model_draws = np.random.multinomial(
            ndraws,
            list(self.posteriors.values())
        )

        draws = []
        for i in range(len(model_draws)):
            
            if model_draws[i] == 0:
                continue
            
            mask = np.fromstring(
                list(self.posteriors.keys())[i][1:-1],
                dtype=bool,
                sep=" "
            )
            draws += list(linear_regression.LinearModel(
                self.X[:, mask],
                self.y,
                g
            ).residual_dist(model_draws[i]))

        return draws


    def coef_dist(self, ndraws=1000):
        """Draw from the posterior distribution of the regression coefficients.

        Parameters
        ----------
        ndraws : int, default 1000
            number of draws

        Retrurns
        --------
        np.ndarray
            vector of draws
        """

        g = max(self.X.shape[0], self.X.shape[1] ** 2)
        model_draws = np.random.multinomial(
            ndraws,
            list(self.posteriors.values())
        )

        draws = []
        for i in range(len(model_draws)):
            
            if model_draws[i] == 0:
                continue
            
            mask = np.fromstring(
                list(self.posteriors.keys())[i][1:-1],
                dtype=bool,
                sep=" "
            )
            draws += list(linear_regression.LinearModel(
                self.X[:, mask],
                self.y,
                g
            ).coef_dist(model_draws[i]))

        return draws
    

    def predictive_dist(self, x_new, ndraws=1000):
        """Draw from the predictive distribution of a new observation.

        Parameters
        ----------
        x_new : np.ndarray
            prediction vector
        ndraws : int, default 1000
            number of draws

        Retrurns
        --------
        np.ndarray
            vector of draws
        """

        g = max(self.X.shape[0], self.X.shape[1] ** 2)
        model_draws = np.random.multinomial(
            ndraws,
            list(self.posteriors.values())
        )

        draws = []
        for i in range(len(model_draws)):
            
            if model_draws[i] == 0:
                continue
            
            mask = np.fromstring(
                list(self.posteriors.keys())[i][1:-1],
                dtype=bool,
                sep=" "
            )
            draws += list(linear_regression.LinearModel(
                self.X[:, mask],
                self.y,
                g
            ).predictive_dist(x_new[mask], model_draws[i]))

        return draws
        

    def _estimate(self, penalty_par):
        """Compute point estimates of the model parameters.

        Parameters
        ----------
        penalty_par : float (0, inf)
            dimensionality penalty
        """
        
        self.estimates = {
            "coefficients": np.zeros(self.X.shape[1] + 1),
            "res_precision": 0
        }
        
        for model, posterior in self.posteriors.items():
            
            mask = np.fromstring(model[1:-1], dtype=bool, sep=" ")
            estimates = linear_regression.LinearModel(
                self.X[:, mask],
                self.y,
                penalty_par
            ).estimates
            
            self.estimates["coefficients"][np.hstack((True, mask))] += posterior * estimates["coefficients"]
            self.estimates["res_precision"] += posterior * estimates["res_precision"]

            

class LinearMC3(core.MC3, LinearEnumerator):
    """Computes the posterior probability distribution over the space of linear regression models.
    
    Suitable for high-dimensional models.
    
    Parameters
    ----------
    X : np.ndarray (nobs x ndim)
        prediction matrix
    y : np.ndarray (nobs x 1)
        response vector
    penalty_par : float (0, inf)
        dimensionality penalty ("g")
    incl_par : float (0, 1)
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
    posteriors: Counter
        posterior distribution over the model space where
        str(model) is the key and the posterior probability is the value
    estimates: dict
        point estimates of the model parameters including
        "coefficients" and "res_precision"
    par : dict
        MCMC parameters including "niter", "proposal"
    diagnostics : dict
        MCMC diagnostics per coefficient including
        "ess" (effective sample size)
        "ndiscard" (pre-equilibrium samples),
        "stderr" (standard error)
    """

    def __init__(self, X, y, penalty_par, incl_par, niter=100000, proposal="random"):

        if int(niter) < 1:  
            raise InputError(
                core.incl_par,
                "incl_par is restricted to {1, .., inf}."
            )
        if not proposal in ("random", "prior"):
            raise InputError(
                core.proposal,
                "proposal is restricted to {'random', 'prior'}."
            )
        
        self.par = {
            "niter": int(niter),
            "proposal": proposal == "random" and self._random_update or self._prior_update
        }
        
        self.X = X
        self.y = y
        self.nobs, self.ndim = X.shape
        self.likelihood_func = lambda X, y: linear_likelihood(X, y, penalty_par)
        self.prior_func = lambda k, n: binom_prior_pmf(k, n, incl_par)

        self._select()
        self._estimate()
