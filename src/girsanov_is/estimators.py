"""Monte Carlo estimators for the European call and the digital call: plain
Monte Carlo and antithetic variates (Section 2 of the note), and importance
sampling with a constant Girsanov drift (Section 3).

Every estimator draws n standard normals Z_i, turns each of them into a
discounted payoff Y_i and averages. Plain Monte Carlo reads Z as
W~_T / sqrt(T), so that S_T is eq. (5.26) of the thesis. Antithetic variates
use the first n/2 draws and their negatives, and average the pair means
(Definition 2.3). Importance sampling reads Z as B_T / sqrt(T), so that S_T
follows eq. (14) of the note, and multiplies each payoff by the weight
dQ/dQ^theta of eq. (16). The result is unbiased for every theta
(Proposition 3.9), and for theta = 0 it is plain Monte Carlo on the same
draws.

Two choices are not forced by the mathematics. Why do the estimators receive
the vector Z, and not a seed? So that all of them can be run on the same
draws, the common random numbers of Section 2.3, on which ``vrf`` compares
them. And why is S_T computed only in ``terminal_price``? One might be
tempted to inline a one-line formula, but theta = 0 has to reproduce plain
Monte Carlo bit for bit, and two formulas that are equal on paper need not be
equal in floating point.
"""

from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.stats import norm


def standard_normals(n, seed):
    """Return ``n`` i.i.d. standard normals from ``seed`` (NumPy's PCG64).

    The same seed gives the same vector on any machine. The estimators never
    see a seed, only this vector.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    return np.random.default_rng(seed).standard_normal(int(n))


def terminal_price(Z, S0, r, sigma, T, theta=0.0):
    """Terminal price S_T from the standard normals ``Z``.

    With theta = 0 this is S_T under Q, eq. (5.26) of the thesis, and with
    theta != 0 it is S_T under Q^theta, eq. (14) of the note, which amounts
    to shifting Z to Z + theta sqrt(T). Both estimators must compute S_T
    here and only here (see the module docstring).
    """
    b = sigma * sqrt(T)  # the b of Section 4 of the note
    return S0 * np.exp((r - 0.5 * sigma**2) * T + b * (Z + theta * sqrt(T)))


def call_payoff(S_T, K):
    """(S_T - K)^+, the payoff of the European call, eq. (5.24) of the thesis."""
    return np.maximum(S_T - K, 0.0)


def digital_payoff(S_T, K):
    """1{S_T > K}, the payoff of the digital of Section 2.2 (strict, as in the
    note, although S_T = K has probability zero)."""
    return (S_T > K).astype(float)


@dataclass(frozen=True)
class MCResult:
    """A Monte Carlo estimate with its error (Section 2.1 of the note).

    estimate      sample mean of the discounted payoffs, eq. (2)
    std_error     s_n / sqrt(n), the estimate of SE_n in eq. (3)
    half_width    z s_n / sqrt(n), with z = N^{-1}(0.5 + confidence / 2)
    ci            (estimate - half_width, estimate + half_width), eq. (4)
    n             number of samples
    variance      s_n^2, the sample variance of one payoff (not of the
                  estimate, which is this divided by n)
    """

    estimate: float
    std_error: float
    half_width: float
    ci: tuple
    n: int
    variance: float


def mc_estimate(Y, confidence=0.95):
    """Summarise a sample ``Y`` of discounted payoffs as an ``MCResult``.

    The interval is the CLT one of eq. (4), so it is only as good as that
    approximation: if no sample ends in the money, s_n = 0 and the interval
    collapses to [0, 0], which does not contain the price (Subsection 6.1 of
    the note).
    """
    Y = np.asarray(Y, dtype=float)
    n = Y.size
    if n < 2:
        raise ValueError("at least 2 samples are needed to estimate the error")
    V_hat = float(Y.mean())
    s2 = float(Y.var(ddof=1))  # the n - 1 of s_n^2 (footnote in Section 2.1)
    SE = sqrt(s2 / n)
    z = norm.ppf(0.5 + confidence / 2)
    half_width = z * SE
    return MCResult(V_hat, SE, half_width, (V_hat - half_width, V_hat + half_width), n, s2)


def mc_call(Z, S0, K, r, sigma, T, confidence=0.95):
    """Plain Monte Carlo estimate of the call price, eq. (2) of the note."""
    S_T = terminal_price(Z, S0, r, sigma, T)
    return mc_estimate(np.exp(-r * T) * call_payoff(S_T, K), confidence)


def mc_digital(Z, S0, K, r, sigma, T, confidence=0.95):
    """Plain Monte Carlo estimate of the digital price, eq. (2) of the note."""
    S_T = terminal_price(Z, S0, r, sigma, T)
    return mc_estimate(np.exp(-r * T) * digital_payoff(S_T, K), confidence)


def likelihood_ratio(Z, theta, T):
    """The weight w_theta(Z) = dQ/dQ^theta of eq. (16) of the note.

    Z is read as B_T / sqrt(T), not as W~_T / sqrt(T). Written in W~_T the
    quadratic term would carry a plus sign, and mixing up the two readings
    is the natural mistake here (the test that the weight has mean one
    catches it).
    """
    a = theta * sqrt(T)  # the a of Section 4 of the note
    return np.exp(-a * Z - 0.5 * a * a)


def is_call(Z, S0, K, r, sigma, T, theta, confidence=0.95):
    """Importance-sampling estimate of the call price with drift ``theta``.

    The terms are the Y_i^theta of Definition 3.8, and with theta = 0 the
    result is exactly that of ``mc_call``. ``theta`` has to be fixed before
    the draws, or come from an independent pilot run (Remark 3.10).
    """
    S_T = terminal_price(Z, S0, r, sigma, T, theta)
    Y = likelihood_ratio(Z, theta, T) * np.exp(-r * T) * call_payoff(S_T, K)
    return mc_estimate(Y, confidence)


def is_digital(Z, S0, K, r, sigma, T, theta, confidence=0.95):
    """Importance-sampling estimate of the digital price with drift ``theta``
    (the remarks on ``is_call`` apply)."""
    S_T = terminal_price(Z, S0, r, sigma, T, theta)
    Y = likelihood_ratio(Z, theta, T) * np.exp(-r * T) * digital_payoff(S_T, K)
    return mc_estimate(Y, confidence)


def _antithetic_pairs(Z):
    """The first half of an even-length ``Z`` and its negative, the two members
    of the antithetic pairs (Definition 2.3 of the note)."""
    Z = np.asarray(Z, dtype=float)
    if Z.size % 2:
        raise ValueError(f"antithetic variates need an even n, got {Z.size}")
    Z_half = Z[: Z.size // 2]
    return Z_half, -Z_half


def av_call(Z, S0, K, r, sigma, T, confidence=0.95):
    """Antithetic-variates estimate of the call price with n = len(Z) payoff
    evaluations (Definition 2.3 of the note).

    The i.i.d. replications are the n/2 pair means (g(Z_i) + g(-Z_i)) / 2, and
    not the n payoffs, which are correlated within each pair. That is why
    ``mc_estimate`` is applied to the pair means, which gives the standard
    error of the paragraph after Proposition 2.4.
    """
    Z_plus, Z_minus = _antithetic_pairs(Z)
    discount = np.exp(-r * T)
    g_plus = discount * call_payoff(terminal_price(Z_plus, S0, r, sigma, T), K)
    g_minus = discount * call_payoff(terminal_price(Z_minus, S0, r, sigma, T), K)
    return mc_estimate(0.5 * (g_plus + g_minus), confidence)


def av_digital(Z, S0, K, r, sigma, T, confidence=0.95):
    """Antithetic-variates estimate of the digital price (the remarks on
    ``av_call`` apply)."""
    Z_plus, Z_minus = _antithetic_pairs(Z)
    discount = np.exp(-r * T)
    g_plus = discount * digital_payoff(terminal_price(Z_plus, S0, r, sigma, T), K)
    g_minus = discount * digital_payoff(terminal_price(Z_minus, S0, r, sigma, T), K)
    return mc_estimate(0.5 * (g_plus + g_minus), confidence)


def vrf(plain, other):
    """Estimated variance reduction factor of ``other`` with respect to the
    plain Monte Carlo result ``plain``, eq. (8) of the note.

    Both results must come from the same n payoff evaluations on the same
    draws. Each standard error already counts its own i.i.d. replications
    (the n/2 pair means, for antithetic variates), so no further bookkeeping
    is needed.
    """
    return (plain.std_error / other.std_error) ** 2
