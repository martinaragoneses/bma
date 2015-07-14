Bayesian Model Averaging
========================

Provides routines for Bayesian Model Averaging (BMA). BMA searches a model space (e.g. linear regression models) for promising models and computes the posterior probability distribution over that space. Coefficients are then estimated from a weighted average over the model space.

Running BMA is as simple as fitting a regression model. Estimates will be close to the ones you would obtain from fitting the "true" nested model, and no knowledge of that model is required.


TOC
---

The following scripts are relevant for end users:
- `linear_regression.py` contains routines for Bayesian linear regression.
- `linear_averaging.py` contains routines for linear BMA.
- `sim.py` demonstrates basic usage of linear BMA.

The following scripts are useful if you wish to adapt BMA to other model spaces:
- `core.py` contains routines for generic BMA.
- `mcmc.py` contains generic MCMC routines.


Usage
-----

The specific Bayesian regression model I use expects 2 hyperparameters:
- *g* is a parameter that penalizes model size. I recommend setting it to max(n_obs, n_dim^2).
- *p* is your prior expectation of how many relevant variables your dataset contains. If you expected 10% of the variables in *X* to be relevant, you would set it to 1/10.

Basic usage is demonstrated in `sim.py`. Given regressors *X* and response *y* You can fit the model by executing

```python
mod = linear_averaging.LinearMC3(X, y, g, p)

mod.select()

mod.estimate()
```

The first step computes the posterior model distribution, the second computes the posterior distributions over the model parameters.

Please consult the docstrings for further documentation.


Dependencies
-------------

All scripts were written with Python 3 in mind and require the usual set of scientific Python libraries. They can be converted to Python 2.7 with minimal changes. It is crucial to enable *true division* by adding the following line to all scripts:

```python
from __future__ import division
```


References
----------

- Kass and Wassermann (1995) and Kass and Raftery (1995) for Bayesian model averaging and MC3.
- Liang et al. (2008) for the Bayesian Linear Regression model.
- Hastings (1970) for details on the Metropolis-Hastings algorithm.
- Sokal (1997) for MCMC diagnostics.