import numpy as np
from scipy.stats import gamma, norm, multivariate_normal, gaussian_kde
import matplotlib.pyplot as plt

from adaptive_mcmc_1d import AdaptiveOptimalScaling_MH_1d
from adaptive_mcmc import AdaptiveOptimalScaling_MH


np.random.seed(51029841)

### 1d example ###

xgrid = np.linspace(-1, 8, 100)
dist = gamma(a=2, scale=1)  # target distribution
target = lambda x: dist.logpdf(x)[0]


niter = 10000
mcmc = AdaptiveOptimalScaling_MH_1d(target)
draws = mcmc.run(niter, burn_in=niter // 2)
mcmc.plot_results(draws, xgrid)


### nd example ###

dim = 50

# construct covariance matrx
M = np.random.randn(dim, dim)
cov_mat = M @ M.T
np.fill_diagonal(cov_mat, cov_mat.diagonal() * 1.01)
# cov_mat /= 50

plt.title("covariance matrix")
plt.imshow(cov_mat)
plt.show()

# target distribution
dist = multivariate_normal(np.zeros(dim), cov=cov_mat)
target = lambda x: dist.logpdf(x)

# run mcmc
niter = 100000
mcmc = AdaptiveOptimalScaling_MH(target, dim)
draws = mcmc.run(niter, burn_in=niter // 10)

### plotting ###
fig, axs = plt.subplots(2, 2, figsize=(15, 7))
fig.suptitle("Marginal distributions")
for i, ax in enumerate(axs.flatten()):

    std = np.sqrt(cov_mat.diagonal()[i])
    x = np.linspace(-3 * std, 3 * std, 500)

    # kde of mcmc draws
    kde = gaussian_kde(draws[:, i])
    kde.set_bandwidth(kde.factor * 1.5)

    ax.plot(x, np.exp(kde.logpdf(x)))
    ax.plot(x, np.exp(norm.logpdf(x, loc=0, scale=std)))
    ax.set_yticks([])

fig.legend(["MCMC", "truth"])
fig.tight_layout()
plt.show()
