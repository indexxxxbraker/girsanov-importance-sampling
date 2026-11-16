"""The parameters of every experiment of the note, in one place.

The figures and tables are produced by ``make_figures.py`` from the values
below and from nothing else, so changing a number here changes the results,
and no number is hard-coded anywhere else. As for the random numbers, one
master seed gives, through ``np.random.SeedSequence.spawn``, an independent
stream for each experiment, and within an experiment every estimator receives
the same vector Z, so that the comparison is made on the same draws (the
common random numbers of Section 2.3 of the note).
"""

from dataclasses import dataclass

import numpy as np

# The model: Black-Scholes under Q, with the running example of the portfolio.
S0 = 100.0
R = 0.05
SIGMA = 0.2
T = 1.0

MASTER_SEED = 20261218

# Names of the experiments, in the order their seeds are spawned. The order is
# part of the configuration, since inserting a name changes the streams of the
# experiments that come after it.
EXPERIMENTS = ("E1", "E2", "E3", "E4", "E5")


def seeds():
    """One independent SeedSequence per experiment, in a dict by name."""
    children = np.random.SeedSequence(MASTER_SEED).spawn(len(EXPERIMENTS))
    return dict(zip(EXPERIMENTS, children))


@dataclass(frozen=True)
class E1:
    """The variance reduction factor against the moneyness K/S0."""

    moneyness: tuple = tuple(round(1.0 + 0.1 * i, 1) for i in range(11))
    n: int = 10**6


@dataclass(frozen=True)
class E2:
    """The variance reduction factor against theta, at one strike."""

    moneyness: float = 1.5
    # a = theta sqrt(T) from -0.5 to 5.5, far enough on each side to show where
    # the drift makes importance sampling worse than plain Monte Carlo.
    a_grid: tuple = tuple(round(-0.5 + 0.05 * i, 2) for i in range(121))
    n: int = 10**5


@dataclass(frozen=True)
class E3:
    """The estimate and its confidence interval against n, deep out of the
    money."""

    payoff: str = "digital"
    moneyness: float = 2.0
    n_grid: tuple = tuple(int(round(10 ** (2.0 + 0.5 * i))) for i in range(9))


@dataclass(frozen=True)
class E4:
    """The summary table, with the estimate, the VRF, the relative cost and the
    efficiency of each estimator (Section 2.3 of the note)."""

    moneyness: tuple = (1.0, 1.2, 1.5, 2.0)
    payoffs: tuple = ("call", "digital")
    n: int = 10**6
    timing_repeats: int = 5       # each timing is the median of this many runs


@dataclass(frozen=True)
class E5:
    """The optimal drift estimated by a pilot run under the heuristic drift,
    against the size of the pilot (Subsection 6.5 of the note)."""

    moneyness: float = 1.5        # the strike of E2
    pilots: int = 500             # independent pilot runs, for the spread of a-hat
    # m from 10^1.5 to 10^4 in half decades. Under theta_h each sample is in
    # the money with probability 1/2, so a pilot has none, and gives no drift,
    # with probability 2^-m. Starting at m = 10, one of the 500 pilots would
    # have none with probability 0.39, and starting at m = 32, with 10^-7.
    m_grid: tuple = tuple(int(round(10 ** (1.5 + 0.5 * i))) for i in range(6))


CONFIDENCE = 0.95
