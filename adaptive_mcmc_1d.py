import numpy as np
from numpy import ndarray
from numpy.random import rand, randn

from scipy.stats import norm

from time import time
from functools import cached_property
from typing import Callable, Optional


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

mpl.rcParams["font.size"] = 20


class AdaptiveOptimalScaling_MH_1d:
    """
    An 1-d implementation of:
        Adaptive optimal scaling of Metropolis–Hastings algorithms using the Robbins–Monro process, Garthwaite et al. (2014),
        https://arxiv.org/abs/1006.3690

    This is the 1d case.
    Based on R code from: https://web.maths.unsw.edu.au/~yanan/RM.html
    """

    def __init__(
        self,
        target: Callable,
        pstar: Optional[float] = None,
        x0: Optional[ndarray] = None,
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
            it is chosen as optimal, for 1d case is 0.44.
        x0:
            Starting position of the chain.

        Returns
        -------
        None.
        """

        self.target = target

        if pstar is None:
            pstar = 0.44

        self._pstar = pstar
        self._n0 = round(5 / (self.pstar * (1 - self.pstar)))

        if x0 is None:
            x0 = np.ones(1)

        # saving chain and theta
        self.draws = [x0]
        self.thetas = [
            np.log(1)  # log scaling constant for RW-MH, i.e. theta = ln(sigma)
        ]

    @property
    def n0(self):
        return self._n0

    @property
    def pstar(self):
        return self._pstar

    @cached_property
    def c(self):
        """Optimal step-size for 1-d case."""
        return 1 / (self.pstar * (1 - self.pstar))

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

        A = []  # keeping track of acceptance rates

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
            prop = crnt + np.exp(theta) * randn()

            # evaluate numerator
            top = self.target(prop, **target_kwargs)

            # compute ratio
            p = min(1, np.exp(top - bottom))
            if rand() < p:
                # accept proposed sample
                crnt = prop
                bottom = top

            num_iter += 1

            ### updating sigma using RM search ###
            if i > self.n0:
                theta += self.c * (p - self.pstar) / (i + self.n0)

                ## checking conditions to restart search ##
                if (i <= imax) and ((num_big < 5) | (num_small < 5)):
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
            A.append(p)

            i += 1

        draws = np.stack(self.draws[-niter + burn_in :])
        print("optimal p* = 0.44,", f"estimated p*: {round(sum(A) / len(A), 3)}")
        return draws[::thin]

    def plot_results(self, draws: ndarray, xgrid: Optional[ndarray] = None):

        if xgrid is None:
            xgrid = np.linspace(-5, 5, 200)

        fig = plt.figure(figsize=(20, 15))
        # fig.suptitle("Controlling spacing around and between subplots", fontsize=25, y=0.96)

        gs1 = GridSpec(2, 2, width_ratios=(1, 1))

        ax1 = fig.add_subplot(gs1[0, :])
        ax1.set_title("Distribution")
        # ax1.set_title(r'Gamma($\alpha$=2, $\theta$ = 1) distribution')
        ax1.hist(draws, bins="sturges", edgecolor="k", density=True, label="MCMC")
        ax1.plot(
            xgrid,
            [np.exp(self.target(np.array([x]))) for x in xgrid],
            linewidth=3,
            label="density",
        )
        ax1.legend()

        ax2 = fig.add_subplot(gs1[-1, :-1])
        ax2.set_title("Scaling ($\sigma$) estimates")
        ax2.plot(np.exp(self.thetas))

        ax3 = fig.add_subplot(gs1[-1, -1])
        ax3.set_title("MCMC draws")
        ax3.plot(self.draws)

        fig.tight_layout()
        plt.show()
