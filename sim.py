"""
Contains utilities that implement the simulation design and generate the figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import selection
import diagnostics

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



# fig 1
x = range(0, 31)
q_std = [1 - np.sum(i) / 30 for i in y]

plt.xlabel("$d_{\gamma}$")
plt.ylabel("$\pi (d_{\gamma} + 1 | d_{\gamma})$")
plt.plot(x, q_std, color="#377eb8")
for p in (1/4, 1/2, 3/4):
    q_new = [
        selection.binom_pmf(i + 1, 30, p) / (selection.binom_pmf(i + 1, 30, p) + selection.binom_pmf(i - 1, 30, p))
        for i in y
    ]
    plt.plot(x, q_new, color="#e41a1c")
plt.show()

    
# fig2a
np.random.seed(2015)
X, y = simulate_data_d15(100)

np.random.seed(2015)
sampler1 = selection.LinearMC3(X, y, proposal="random", penalty_par=30**2, incl_par=1/3, niter=100000)
diags1 = diagnostics.MC2Diagnostics(sampler1.draws)

plt.xlabel("$t$")
plt.ylabel("$\kappa (t)$")
for acf in diags1.acfs[:10]:
    plt.plot(range(1, len(acf) + 1), acf, color="#377eb8", alpha=0.5)
for acf in diags1.acfs[10:]:
    plt.plot(range(1, len(acf) + 1), acf, color="#e41a1c", alpha=0.5)
plt.show()


# fig2b
np.random.seed(2015)
X, y = simulate_data_d15(100)

np.random.seed(2015)
sampler2 = selection.LinearMC3(X, y, proposal="prior", penalty_par=30**2, incl_par=1/3, niter=100000)
diags2 = diagnostics.MC2Diagnostics(sampler2.draws)

plt.xlabel("$t$")
plt.ylabel("")
for acf in diags2.acfs[:10]:
    plt.plot(range(1, len(acf) + 1), acf, color="#377eb8", alpha=0.5)
for acf in diags2.acfs[10:]:
    plt.plot(range(1, len(acf) + 1), acf, color="#e41a1c", alpha=0.5)
plt.show()


# fig3a
np.random.seed(2015)
X, y = simulate_data_d30(100)

np.random.seed(2015)
ess_array3 = replicate_trial(
    lambda: selection.LinearMC3(X, y, proposal="random", penalty_par=30**2, incl_par=1/3, niter=20000).diagnostics["ess"],
    10
)

plt.ylim(0, 2000)
plt.xlabel("$\gamma_i$")
plt.ylabel("ESS")
fig3 = plt.boxplot(ess_array3)
plt.xticks(range(5, 31, 5), range(5, 31, 5))
plt.setp(fig3['boxes'], color='#444444')
plt.setp(fig3['whiskers'], color='#444444')
plt.setp(fig3['fliers'], color='#e41a1c', marker='+')
plt.show()


# fig3b
np.random.seed(2015)
X, y = simulate_data_d30(100)

np.random.seed(2015)
ess_array4 = replicate_trial(
    lambda: selection.LinearMC3(X, y, proposal="prior", penalty_par=30**2, incl_par=1/3, niter=20000).diagnostics["ess"],
    10
)

plt.ylim(0, 2000)
plt.xlabel("$\gamma_i$")
plt.ylabel("")
fig4 = plt.boxplot(ess_array4)
plt.xticks(range(5, 31, 5), range(5, 31, 5))
plt.setp(fig4['boxes'], color='#444444')
plt.setp(fig4['whiskers'], color='#444444')
plt.setp(fig4['fliers'], color='#e41a1c', marker='+')
plt.show()


# fig4a
np.random.seed(2015)
X, y = simulate_data_d30(100)

np.random.seed(2015)
sampler5 = selection.LinearMC3(X, y, proposal="random", penalty_par=30**2, incl_par=1/3, niter=20000)

plt.xlabel("$\gamma_i$")
plt.ylabel("$\pi (\gamma_i | \mathbf{y})$")
plt.ylim(0, 1)
plt.errorbar(
    range(1, 31),
    sampler5.test_single_coefficients(),
    fmt="o",
    color="#444444",
    alpha=0.5,
    yerr=1.96*sampler5.diagnostics["stderr"],
    ecolor="#e41a1c"
)
plt.show()


# fig4b
np.random.seed(2015)
X, y = simulate_data_d30(100)

np.random.seed(2015)
sampler6 = selection.LinearMC3(X, y, proposal="prior", penalty_par=30**2, incl_par=1/3, niter=20000)

plt.xlabel("$\gamma_i$")
plt.ylabel("")
plt.ylim(0, 1)
plt.errorbar(
    range(1, 31),
    sampler6.test_single_coefficients(),
    fmt="o",
    color="#444444",
    alpha=0.5,
    yerr=1.96*sampler6.diagnostics["stderr"],
    ecolor="#e41a1c"
)
plt.show()
