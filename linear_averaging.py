"""
Provides routines for Bayesian model choice in a linear regression context, such that
    E[y|x_1, ..., x_n] = a + Xb
    where y is the response, b are coefficients and X are predictors.

The full probability model is given by
    p[y, a, b, q, M] = p[y | a, b, q, M] * p[b | q, M] * p[a, q] * p[M]
    where M is the model/hypothesis and q is the residual precision

You may compute Pr[M|y] directly by finding the posterior probability of all possible models. You may also approximate it through MCMC simulation for better scaling.

References
----------
See Liang et al. (2008) for linear model averaging.
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
    


class LinearEnumerator(core.Enumerator):
    """Computes the posterior probability distribution over the space of linear regression models.

    This method computes 2^d probabilities, where d is the number of predictors. Use MC3 for larger d.
    
    Parameters
    ----------
    X : np.ndarray in R^(nobs x ndim)
        predictor matrix
    y : np.ndarray in R^nobs
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
    posterior: Counter
        posterior distribution over the model space where
        str(model) is the key and the posterior probability is the value
    estimates: dict
        point estimates of the model parameters including
        "coefficients" and "res_precision"
    """

    def __init__(self, X, y, penalty_par, incl_par):

        self.nobs, self.ndim = X.shape
        self.X = X
        self.y = y
        self.par = {"penalty": penalty_par, "incl": incl_par}    

        
    def estimate(self):
        """Compute point estimates of the model parameters.
        """
        
        self.estimates = {
            "coefficients": np.zeros(self.X.shape[1] + 1),
            "res_precision": 0
        }
        
        for model, weight in self.posterior.items():
            
            mask = np.fromstring(model[1:-1], dtype=bool, sep=" ")
            model = linear_regression.LinearModel(
                self.X[:, mask],
                self.y,
                self.par["penalty"]
            )
            model.estimate()
            
            self.estimates["coefficients"][np.hstack((True, mask))] += weight * model.estimates["coefficients"]
            self.estimates["res_precision"] += weight * model.estimates["res_precision"]
        

    def predict(self, X_new):
        """Give point predictions for a new set of observations.

        Parameters
        ----------
        X_new : np.ndarray in R^(nobs x ndim)
            prediction matrix

        Returns
        -------
        np.ndarray
            vector of point estimates
        """

        design = np.hstack((np.ones((X_new.shape[0], 1)), X_new))
        return np.dot(design, self.estimates["coefficients"])


    def residual_dist(self, ndraws=1000):
        """Draw from the posterior distribution of the variance of residuals.

        Parameters
        ----------
        ndraws : int {1, .., inf}, default 1000
            number of draws

        Retrurns
        --------
        np.ndarray
            vector of draws
        """

        model_draws = np.random.multinomial(
            ndraws,
            list(self.posterior.values())
        )

        draws = np.empty(0)
        for i, ndraws in enumerate(model_draws):
            
            if ndraws == 0:
                continue
            
            mask = np.fromstring(
                list(self.posterior.keys())[i][1:-1],
                dtype=bool,
                sep=" "
            )
            
            model = linear_regression.LinearModel(
                self.X[:, mask],
                self.y,
                self.par["penalty"]
            )
            model.estimate()
            
            draws = np.append(draws, model.residual_dist(ndraws))

        return draws
    

    def predictive_dist(self, x_new, ndraws=1000):
        """Draw from the predictive distribution of a new observation.

        Parameters
        ----------
        x_new : np.ndarray in R^ndim
            prediction vector
        ndraws : int {1, .., inf}, default 1000
            number of draws

        Retrurns
        --------
        np.ndarray
            vector of draws
        """

        model_draws = np.random.multinomial(
            ndraws,
            list(self.posterior.values())
        )

        draws = np.empty(0)
        for i, ndraws in enumerate(model_draws):
            
            if ndraws == 0:
                continue
            
            mask = np.fromstring(
                list(self.posterior.keys())[i][1:-1],
                dtype=bool,
                sep=" "
            )
            
            model = linear_regression.LinearModel(
                self.X[:, mask],
                self.y,
                self.par["penalty"]
            )
            model.estimate()
            
            draws = np.append(draws, model.predictive_dist(x_new[mask], ndraws))

        return draws


    def _get_prior_prob(self, k, n):
        """Compute the prior probability of a model with k variables.

        Parameters
        ----------
        k : int {1, ..., n}
            number of variables in the model
        n : int {1, ..., inf}
            total number of predictors

        Returns
        -------
        float
            prior probability
        """

        if k < 0 or n < k:
            return 0
        return self.par["incl"] ** k * (1 - self.par["incl"]) ** (n - k)


    def _get_likelihood(self, model):
        """Compute the marginal likelihood of the linear model with a g-prior on betas.

        Parameters
        ----------
        model : np.ndarray in R^ndim
            vector of variable inclusion indicators

        Returns
        -------
        float
            log marginal likelihood
        """

        X = self.X[:, model == 1]
        y = self.y
        nobs, ndim = X.shape
        design = np.hstack((np.ones((nobs, 1)), X))

        mle = np.linalg.solve(np.dot(design.T, design), np.dot(design.T, y))
        residuals = y - np.dot(design, mle)
        rsquared = 1 - np.var(residuals) / np.var(y)

        return (log_gamma((nobs - 1) / 2)
            - (nobs - 1) / 2 * np.log(np.pi)
            - 0.5 * np.log(nobs)
            - (nobs - 1) / 2 * np.log(np.dot(residuals, residuals))
            + (nobs - ndim - 1) / 2 * np.log(1 + self.par["penalty"])
            - (nobs - 1) / 2 * np.log(1 + self.par["penalty"] * (1 - rsquared)))



class LinearMC3(core.MC3, LinearEnumerator):
    """Computes the posterior probability distribution over the space of linear regression models.
    
    Suitable for high-dimensional models.
    
    Parameters
    ----------
    X : np.ndarray in R^(nobs x ndim)
        predictor matrix
    y : np.ndarray in R^nobs
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
    posterior: Counter
        posterior distribution over the model space where
        str(model) is the key and the posterior probability is the value
    estimates: dict
        point estimates of the model parameters including
        "coefficients" and "res_precision"
    """

    def __init__(self, X, y, penalty_par, incl_par):
        
        # wrap parent constructor
        LinearEnumerator.__init__(self, X, y, penalty_par, incl_par)
