"""
isort:skip_file
"""
import argparse
import os
import pickle

import sys

sys.path.append("/home/tomas/ssm/")

import matplotlib.pyplot as plt
import numpy as np

from scipy import stats

from ssm.mcmc import (
    Distribution,
    MetropolisHastingsABC,
    MetropolisHastingsPF,
    Prior,
    Proposal,
)
from ssm.utils import check_random_state


parser = argparse.ArgumentParser()
parser.add_argument(
    "--algorithm",
    choices=("abcmh", "pmh"),
    required=True,
    help="The algorithm to run. Either ABC Metropolis-Hastings or Particle Metropolis-Hastings.",
)
parser.add_argument(
    "--n-samples",
    type=int,
    required=True,
    help="Number of Metropolis-Hastings samples.",
)
parser.add_argument(
    "--n-particles", type=int, required=True, help="Number of particles."
)
parser.add_argument(
    "--burn-in", type=int, default=0, help="Length of the burn-in period."
)
parser.add_argument(
    "--thinning", type=int, default=1, help="Thinning of the Markov chain."
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.9,
    help="Fraction of pseudo-observations covered by the kernel. Only applies if algorithm=abcmh.",
)
parser.add_argument(
    "--hpr-p",
    type=float,
    default=0.95,
    help="Width of the p-HPR of the kernel. Only applies if algorithm=abcmh.",
)
parser.add_argument(
    "--kernel",
    choices=("gaussian", "cauchy", "uniform"),
    default="gaussian",
    help="Kernel type.",
)
args = parser.parse_args()


class ABCLorenzSystem(MetropolisHastingsABC):
    def _transition(self, x: np.ndarray, t: int, theta: np.ndarray) -> np.ndarray:
        observation_period = self.const["observation_period"]
        T = self.const["T"]
        sqrt_T = np.sqrt(T)
        S = theta[0]
        R = theta[1]
        B = theta[2]

        for _ in range(observation_period):
            out = np.empty_like(x)
            U = self.random_state.normal(size=x.shape)
            out[:, 0] = x[:, 0] - T * S * (x[:, 0] - x[:, 1]) + sqrt_T * U[:, 0]
            out[:, 1] = (
                x[:, 1] + T * (R * x[:, 0] - x[:, 1] - x[:, 0] * x[:, 2]) + sqrt_T * U[:, 1]
            )
            out[:, 2] = x[:, 2] + T * (x[:, 0] * x[:, 1] - B * x[:, 2]) + sqrt_T * U[:, 2]
            x = out

        assert out.shape == x.shape
        return out

    def _measurement_model(self, x: np.ndarray, theta: np.ndarray) -> np.array:
        k = theta[3]
        var = self.const["observation_variance"]

        mean0 = k * x[:, 0]
        mean2 = k * x[:, 2]

        mean = np.c_[mean0, mean2]
        assert mean.shape == (self.n_particles, 2)

        return mean


class ParticleLorenzSystem(MetropolisHastingsPF):
    def _transition(self, x: np.ndarray, t: int, theta: np.ndarray) -> np.ndarray:
        observation_period = self.const["observation_period"]
        T = self.const["T"]
        sqrt_T = np.sqrt(T)
        S = theta[0]
        R = theta[1]
        B = theta[2]

        for _ in range(observation_period):
            out = np.empty_like(x)
            U = self.random_state.normal(size=x.shape)
            out[:, 0] = x[:, 0] - T * S * (x[:, 0] - x[:, 1]) + sqrt_T * U[:, 0]
            out[:, 1] = (
                x[:, 1] + T * (R * x[:, 0] - x[:, 1] - x[:, 0] * x[:, 2]) + sqrt_T * U[:, 1]
            )
            out[:, 2] = x[:, 2] + T * (x[:, 0] * x[:, 1] - B * x[:, 2]) + sqrt_T * U[:, 2]
            x = out

        assert out.shape == x.shape
        return out

    def _observation_log_prob(
        self, y: np.ndarray, x: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        k = theta[3]
        var = self.const["observation_variance"]

        mean0 = k * x[:, 0]
        mean2 = k * x[:, 2]

        mean = np.c_[mean0, mean2]
        assert mean.shape == (self.n_particles, 2)

        #out = stats.multivariate_normal.logpdf(y, mean=mean, cov=var)
        out = np.sum(stats.norm.logpdf(y, loc=mean, scale=np.sqrt(var)), axis=1)
        assert out.shape == (self.n_particles,)

        return out


def simulate_ty(
    path: str,
    n_observations: int,
    S: float,
    R: float,
    B: float,
    k: float,
    x0: np.ndarray,
    T: float,
    observation_period: int,
    observation_variance: float,
    random_state=None,
):
    if os.path.exists(path):
        with open(path, mode="rb") as f:
            return pickle.load(f)
    else:
        sqrt_T = np.sqrt(T)
        observation_std = np.sqrt(observation_variance)
        ts = np.arange(n_observations)
        ys = np.empty(shape=(n_observations, 2), dtype=float)
        x = x0.copy()

        for t in range(1, n_observations + 1):
            for _ in range(observation_period):
                # x_t -> x_{t+1}
                out = np.empty_like(x)
                U = random_state.normal(size=3)
                out[0] = x[0] - T * S * (x[0] - x[1]) + sqrt_T * U[0]
                out[1] = x[1] + T * (R * x[0] - x[1] - x[0] * x[2]) + sqrt_T * U[1]
                out[2] = x[2] + T * (x[0] * x[1] - B * x[2]) + sqrt_T * U[2]
                x = out

            # x_{n(t+1)} -> y_{t+1}, n = observation_period
            V = random_state.normal(scale=observation_std, size=2)
            ys[t - 1, 0] = k * x[0] + V[0]
            ys[t - 1, 1] = k * x[2] + V[1]

        with open(path, mode="wb") as f:
            pickle.dump((ts, ys), f)

        return ts, ys


def main():
    algorithm = args.algorithm
    path = "./lorenz_system_{}".format(algorithm)
    random_state = check_random_state(1)

    if not os.path.exists(path):
        os.makedirs(path)

    # True parameters
    S = 10
    R = 28
    B = 8 / 3
    k = 4 / 5
    theta_true = np.array([S, R, B, k])

    # Initial state
    x_star = np.array([-5.91652, -5.52332, 24.5723])
    v0_squared = 10
    x0 = stats.multivariate_normal.rvs(
        mean=x_star, cov=v0_squared, random_state=random_state
    )

    # Constants
    T = 1e-3
    observation_period = 1
    observation_variance = 1 / 10

    n_observations = 100  # TODO: Set to 600. This means 600 observations*40 period=24000 time steps, as in the paper.

    t, y = simulate_ty(
        os.path.join(path, "simulated_data.pickle"),
        n_observations=n_observations,
        S=S,
        R=R,
        B=B,
        k=k,
        x0=x0,
        T=T,
        observation_period=observation_period,
        observation_variance=observation_variance,
        random_state=random_state,
    )
    n_samples = args.n_samples
    n_particles = args.n_particles
    burn_in = args.burn_in
    thinning = args.thinning

    state_init = x0.copy()

    const = {
        "observation_period": observation_period,
        "observation_variance": observation_variance,
        "times": np.concatenate(([0.0], t)),
        "T": T,
    }

    prior = Prior(
        [
            stats.uniform(loc=5, scale=20 - 5),
            stats.uniform(loc=18, scale=50 - 18),
            stats.uniform(loc=1, scale=8 - 1),
            stats.uniform(loc=0.5, scale=3 - 0.5),
        ]
    )

    scale_S = np.sqrt(60 / np.power(n_particles, 3 / 2))
    scale_R = np.sqrt(60 / np.power(n_particles, 3 / 2))
    scale_B = np.sqrt(10 / np.power(n_particles, 3 / 2))
    scale_k = np.sqrt(1 / np.power(n_particles, 3 / 2))

    proposal = Proposal(
        [
            Distribution(stats.truncnorm, truncnorm=True, scale=scale_S, a=5, b=20),
            Distribution(stats.truncnorm, truncnorm=True, scale=scale_R, a=18, b=50),
            Distribution(stats.truncnorm, truncnorm=True, scale=scale_B, a=1, b=8),
            Distribution(stats.truncnorm, truncnorm=True, scale=scale_k, a=0.5, b=3),
        ]
    )

    theta_init = np.zeros(4)
    scale = 2.0
    theta_init[0] = stats.truncnorm.rvs(a=(5 - theta_true[0]) / scale, b = (20 - theta_true[0]) / scale, loc=theta_true[0], scale=scale, random_state=random_state)
    theta_init[1] = stats.truncnorm.rvs(a=(18 - theta_true[1]) / scale, b = (50 - theta_true[1]) / scale, loc=theta_true[1], scale=scale, random_state=random_state)
    theta_init[2] = stats.truncnorm.rvs(a=(1 - theta_true[2]) / scale, b = (8 - theta_true[2]) / scale, loc=theta_true[2], scale=scale, random_state=random_state)
    theta_init[3] = stats.truncnorm.rvs(a=(0.5 - theta_true[3]) / scale, b = (3 - theta_true[3]) / scale, loc=theta_true[3], scale=scale, random_state=random_state)

    print(theta_true)
    print(theta_init)

    if algorithm == "abcmh":
        alpha = args.alpha
        hpr_p = args.hpr_p
        kernel = args.kernel

        mcmc = ABCLorenzSystem(
            n_samples=n_samples,
            n_particles=n_particles,
            alpha=alpha,
            hpr_p=hpr_p,
            state_init=state_init,
            const=const,
            kernel=kernel,
            prior=prior,
            proposal=proposal,
            theta_init=theta_init,
            random_state=random_state,
        )
    else:
        mcmc = ParticleLorenzSystem(
            n_samples=n_samples,
            n_particles=n_particles,
            state_init=state_init,
            const=const,
            prior=prior,
            proposal=proposal,
            theta_init=theta_init,
            random_state=random_state,
        )

    sampled_theta_path = os.path.join(path, "sampled_theta.pickle")

    if os.path.exists(sampled_theta_path):
        with open(sampled_theta_path, "rb") as f:
            theta = pickle.load(f)
    else:
        theta = mcmc.do_inference(y)

        with open(sampled_theta_path, "wb") as f:
            pickle.dump(theta, f)

    theta = theta[burn_in::thinning]
    pretty_names = [r"$S$", r"$R$", r"$B$", r"$k$"]

    for i in range(theta.shape[1]):
        param_name = pretty_names[i]
        param_values = theta[:, i]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        plt.suptitle(param_name)

        ax1.set_title("Trace plot")
        ax1.plot(param_values, color="dimgrey")
        ax1.axhline(theta_true[i], color="crimson", lw=2)

        # plot_acf(param_values, lags=100, ax=ax2, color="dimgrey")

        ax3.set_title("Histogram")
        ax3.hist(param_values, density=True, bins=30, color="dimgrey")
        ax3.axvline(theta_true[i], color="crimson", lw=2)

        plt.show()


if __name__ == "__main__":
    main()
