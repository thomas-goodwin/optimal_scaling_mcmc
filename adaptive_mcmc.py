import numpy as np
from numpy import ndarray
from numpy.random import rand, randn

from scipy.stats import norm

from time import time
from typing import Callable, Optional


def update_mean_and_Sigma(i: int, x: ndarray, x_bar: ndarray, Sigma_hat: ndarray):
    """
    Updating the mean and covariance of x (as new draws come through).
    Everything written in terms of (i+1)
    """

    assert i != 0, "divide by zero error"

    # previous x_bar
    mid = x_bar[None] * x_bar[:, None]

    x_bar = (i * x_bar + x) / (i + 1)

    Sigma_hat = (
        ((i - 1) / (i)) * Sigma_hat
        + mid
        - ((i + 1) / (i)) * x_bar[None] * x_bar[:, None]
        + (1 / i) * x[None] * x[:, None]
    )  # + (1 / i) * np.eye(len(Sigma_hat))               # adding epsilon * I
    # + (1e-6) * np.eye(len(Sigma_hat))                # adding epsilon * I

    return x_bar, Sigma_hat


class AdaptiveOptimalScaling_MH:
    """
    An implementation of:
        Adaptive optimal scaling of Metropolis–Hastings algorithms using the Robbins–Monro process, Garthwaite et al. (2014),
        https://arxiv.org/abs/1006.3690

    This is the multivariate case.
    Based on R code from: https://web.maths.unsw.edu.au/~yanan/RM.html

    This also implements the adaptive Metropolis algorithm based on Haario et al. (2001), together with Garthwaite et al. (2014)
    """

    def __init__(
        self,
        target: Callable,
        dim: int,
        pstar: Optional[float] = None,
        x0: Optional[ndarray] = None,
        sigma0: float = 1.0,
    ):
        """
        Parameters
        ----------
        target: Callable
            Log-pdf of the target distribution.
        dim: int
            Dimension of the distribution.
        pstar:
            The optimal acceptance probability. If None,
            it is chosen as optimal, for multivariate case is 0.234.
        x0:
            Starting position of the chain.

        Returns
        -------
        None.
        """

        self.target = target
        self.dim = dim

        if pstar is None:
            pstar = 0.234

        self._pstar = pstar

        self.alpha = self.compute_alpha(self.pstar)
        self._n0 = round(5 / (self.pstar * (1 - self.pstar)))

        if x0 is None:
            x0 = np.ones(dim)  # TODO: change this

        # running mean and covariance
        self.x_bar = x0.copy()
        self.Sigma_bar = np.eye(self.dim)

        # saving chain and theta
        self.draws = [x0]
        self.thetas = [
            np.log(sigma0)
        ]  # log scaling constant for RW-MH, i.e. theta = ln(sigma)
        # TODO: make starting thetas0 in arguments!!

    @property
    def n0(self):
        return self._n0

    @property
    def pstar(self):
        return self._pstar

    def compute_alpha(self, pstar: float):
        return -norm.ppf(pstar / 2)  # quantile function

    def update_theta(self, theta: float, p: float, i: int):
        """
        Updates theta = exp(sigma).
        pstar is optimal acceptance rate.
        p is acceptance probability of MCMC at iteration i.
        """

        # compute optimal step-size
        c = (1 - 1 / self.dim) * np.sqrt(2 * np.pi) * np.exp(self.alpha**2 / 2) / (
            2 * self.alpha
        ) + 1 / (self.dim * self.pstar / (1 - self.pstar))

        eps = max(200, i / self.dim)

        # update theta
        # dividing by large number ensures theta doesnt change much
        # after many mcmc iterations, i.e. decreases 'log-linearly'
        theta += c * (p - self.pstar) / eps
        return theta

    def run(
        self,
        niter: int,
        burn_in: int = 0,
        thin: int = 1,
        n_disp: int = 500,
        **target_kwargs,
    ):
        """
        Run MCMC.
        """

        x0 = self.draws[-1]
        theta0 = self.thetas[-1]

        imax = 100  # max number of iterations before the last restart
        num_big = 0
        num_small = 0

        A = []
        # Sigmas = []

        i = 0
        num_iter = 0
        n_restarts = 0

        theta = theta0.copy()
        crnt = x0.copy()  # current step
        bottom = self.target(crnt, **target_kwargs)  # log demonimator of MH ratio

        t0 = time()
        while i < niter:

            if (i + 1) % n_disp == 0:
                print(
                    f"Iteration: {i+1}    Acceptance rate: {np.mean(A[i-(n_disp-1): (i+1)]).round(3)}    Time: {np.round(time()-t0,3)}s"
                )

            ### RW MH step ###
            if i <= 100 - 1:
                prop_cov = np.eye(self.dim)
                # prop = crnt + 2.38 ** 2 randn(dim)

            else:
                prop_cov = np.exp(theta) * self.Sigma_bar

            # propose sample
            prop = np.random.multivariate_normal(crnt, prop_cov)

            # evaluate numerator
            top = self.target(prop, **target_kwargs)

            # compute ratio
            p = min(1, np.exp(top - bottom))
            if rand() < p:
                # accept proposed sample
                crnt = prop
                bottom = top

            num_iter += 1

            ### updating mean and covariance of draws ###
            if i > 1:
                self.x_bar, self.Sigma_bar = update_mean_and_Sigma(
                    i, crnt, self.x_bar, self.Sigma_bar
                )
                self.Sigma_bar += (1 / i) * np.eye(self.dim)

            ### updating theta using RM ###
            if i > self.n0:
                theta = self.update_theta(theta, p, i)

                ## checking conditions to restart search ##
                if (i <= (imax + self.n0)) and ((num_big < 5) | (num_small < 5)):
                    too_big = np.exp(theta) > 3 * np.exp(theta0)
                    too_small = np.exp(theta) < np.exp(theta0) / 3

                    if too_big | too_small:
                        # restart algorithm
                        print(f"Restarting algorithm at {num_iter}th iteration")

                        num_big += too_big  # adding 1 or 0
                        num_small += too_small  # adding 1 or 0
                        i = self.n0
                        theta0 = theta
                        n_restarts += 1

            ### appending results ###
            self.draws.append(crnt)
            self.thetas.append(theta)
            # Sigmas.append(Sigma_bar)
            A.append(p)

            i += 1

        draws = np.stack(self.draws[burn_in:])
        print("optimal p* = 0.234,", f"estimated p*: {round(sum(A) / len(A), 3)}")
        return draws[::thin]


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal
    from whittlex.plotting_funcs import plot_marginals

    # np.random.seed(51029841)

    dim = 50
    M = randn(dim, dim)

    cov_mat = M @ M.T
    np.fill_diagonal(cov_mat, cov_mat.diagonal() * 1.01)
    # cov_mat /= 50
    # print(cov_mat)

    plt.imshow(cov_mat)
    plt.show()

    dist = multivariate_normal(np.zeros(dim), cov=cov_mat)
    target = lambda x: dist.logpdf(x)

    niter = 10000
    mcmc = AdaptiveOptimalScaling_MH(target, dim)
    draws = mcmc.run(niter, burn_in=niter // 2)

    plot_marginals([draws], shape=(5, 10), figsize=(20, 15))
