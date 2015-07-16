"""
Provides routines for Bayesian linear regression, such that
    E[y|x_1, ..., x_n] = a + Xb
    where y is the response, b are coefficients and X are predictors.

The probability model is given by
    p[y, a, b, q] = p[y | a, b, q] * p[b | q] * p[a, q]
    where q is the residual precision.
"""

import numpy as np
from scipy.optimize import brentq



def draw_student(loc, scale, df, ndraws):
    """Draw from the multivariate student-t distribution.

    Parameters
    ----------
    loc : np.ndarray in R^ndim
        vector of means
    scale : np.ndarray in R^(ndim x ndim)
        positive definite scale matrix
    df : int {1, .., inf}
        degrees of freedom
    ndraws : int {1, .., inf}
        number of draws

    Returns
    -------
    np.ndarray
        matrix of draws
    """

    scaling = np.random.gamma(df / 2, 2 / df, ndraws) ** 0.5

    if type(loc) != np.ndarray:
        return loc + np.random.normal(0, scale, ndraws) / scaling

    normal_draws = np.random.multivariate_normal(
        np.zeros(loc.shape[0]),
        scale,
        ndraws
    )

    return loc + normal_draws / scaling[:, np.newaxis]


def draw_normal_gamma(params, ndraws):
    """Draw from the multivariate student-t distribution.

    Parameters
    ----------
    params : dict
        parameters including "shape", "rate", "precision", "location"
    ndraws : int {1, .., inf}
        number of draws

    Returns
    -------
    dict
        including "gamma" and "normal" draws
    """

    gamma_draws = np.random.gamma(
            params["shape"],
            1 / params["rate"],
            ndraws
        )

    if type(params["precision"]) != np.ndarray:
        normal_draws = np.array([
            np.random.normal(
                params["location"],
                1 / params["precision"] / gamma_draws[i]
            )
            for i in range(ndraws)
        ])

    else:
        normal_draws = np.array([
            np.random.multivariate_normal(
                params["location"],
                np.linalg.inv(params["precision"]) / gamma_draws[i]
            )
            for i in range(ndraws)
        ])

    return {"gamma": gamma_draws, "normal": normal_draws}


def integrate(draws, lo, hi):
    """Approximate the probability integral from a series of draws.

    Parameters
    ----------
    draws : np.ndarray in R^nobs or R^(ndim x nobs)
        vector or matrix of draws from a random variable or vector
    lo : float or np.ndarray in R^ndim
        lower bound of the integral
    hi : float or np.ndarray in R^ndim
        upper bound of the integral

    Returns
    -------
    float
        probability integral
    """

    return np.mean(np.logical_and(lo <= draws, draws <= hi))


def get_prediction_interval(draws, alpha=0.05):
    """Approximate the (1 - alpha)-probability interval from a series of draws.

    If you pass a series of multivariate draws, the dimensions will be scaled according to their standard deviation.

    Parameters
    ----------
    draws : np.ndarray in R^nobs or R^(ndim x nobs)
        vector or matrix of draws from a random variable or vector
    alpha : float (0, 1)
        error probability

    Returns
    -------
    tup
        bounds of the prediction interval
    """

    integral = lambda x, mean, std, alpha: alpha - 1 + integrate(
        draws,
        mean - x * std,
        mean + x * std
    )
    mean = np.mean(draws, axis=0)
    std = np.var(draws, axis=0) ** 0.5
    offset = brentq(integral, 1 - alpha, 1 / alpha, args=(mean, std, alpha))

    return (mean - offset * std, mean + offset * std)



class LinearModel(object):
    """Computes the posterior distribution of the models' coefficients.

    Provides routines to draw from relevant posterior distributions.

    Parameters
    ----------
    X : np.ndarray in R^(nobs x ndim)
        predictor matrix
    y : np.ndarray in R^nobs
        response vector
    penalty_par : float (0, inf)
        dimensionality penalty ("g")

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
    posterior: dict
        posterior distribution over the model parameters including
        "shape", "rate", "location", "precision"
    estimates: dict
        point estimate of the model parameters including
        "coefficients" and "res_precision"
    """


    def __init__(self, X, y, penalty_par):

        self.nobs, self.ndim = X.shape
        self.X = X
        self.y = y
        self.penalty_par = penalty_par


    def estimate(self):
        """Compute the parameters of the posterior distribution of the model coefficients.
        """

        design = np.hstack((np.ones((self.nobs, 1)), self.X))
        sub_quadrant = (1 + 1 / self.nobs / self.penalty_par) * np.dot(self.X.T, self.X)

        precision = np.vstack((
            np.hstack((self.nobs, np.sum(self.X, 0).T)),
            np.hstack((np.sum(self.X, 0, keepdims=True).T, sub_quadrant))
        ))
        location = np.linalg.solve(precision, np.dot(design.T, self.y))
        shape = (self.nobs - 1) / 2
        rate = 0.5 * np.dot(self.y - np.dot(design, location), self.y)

        self.posterior = {
            "shape": shape,
            "rate": rate,
            "location": location,
            "precision": precision
        }
        self.estimates = {
            "coefficients": location,
            "res_precision": shape / rate
        }


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

        return np.random.gamma(
            self.posterior["shape"],
            1 / self.posterior["rate"],
            ndraws
        )


    def coef_dist(self, ndraws=1000):
        """Draw from the posterior distribution of the regression coefficients.

        Parameters
        ----------
        ndraws : int {1, .., inf}, default 1000
            number of draws

        Retrurns
        --------
        np.ndarray
            matrix of draws
        """

        return draw_normal_gamma(self.posterior, ndraws)["normal"]


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

        ng_draws = draw_normal_gamma(self.posterior, ndraws)
        res_draws = np.random.randn(ndraws) / ng_draws["gamma"] ** 0.5

        return np.dot(ng_draws["normal"], np.hstack((1, x_new))) + res_draws
