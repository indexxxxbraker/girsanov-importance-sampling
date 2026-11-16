"""The two choices of the drift theta of Section 4 of the note, the
closed-form second moment that compares them, and the estimate of the optimal
drift from a pilot run of Subsection 6.5.

The notation is that of the note: a = theta sqrt(T), b = sigma sqrt(T), and
d_2 and d_1 = d_2 + b as in eqs. (5.27) and (5.28) of the thesis. The
heuristic ``theta_heuristic`` puts the median of S_T under Q^theta at the
strike (Proposition 4.1), so all it needs is where the strike falls in the
law of S_T. The optimal drift, ``theta_optimal``, minimises the second moment
M_2(theta), which ``second_moment`` computes in closed form (Proposition 4.3)
and which is strictly convex and coercive (Theorem 4.4), so the minimiser is
unique. For the digital, ``theta_optimal_digital_root`` reaches the same
drift as the root of eq. (25) (Proposition 4.5), and the tests check the two
against each other.

Note that theta* is an oracle: it needs the closed form of M_2, which exists
only because Black-Scholes is solvable. It is the reference of what a
constant drift can achieve, and the heuristic is measured against it. In a
problem without a closed form theta* would have to be estimated, and
``theta_pilot`` does it as Subsection 6.5 describes: it minimises
``pilot_second_moment``, the estimate (28) of the whole function M_2 from a
single pilot sample. Here the closed form is still at hand to tell how good
that estimate is.
"""

from math import exp, sqrt

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import softmax
from scipy.stats import norm

from girsanov_is.black_scholes import d2
from girsanov_is.estimators import call_payoff, digital_payoff, terminal_price


def theta_heuristic(S0, K, r, sigma, T):
    """The heuristic drift theta_h of eq. (20) of the note, which puts the
    median of S_T under Q^theta at the strike."""
    return -d2(S0, K, r, sigma, T) / sqrt(T)


def second_moment(theta, S0, K, r, sigma, T, payoff):
    """The second moment M_2(theta) of w Y under Q^theta, in closed form, for
    ``payoff`` in {"call", "digital"}: eqs. (22) and (23) of the note.

    ``theta`` may be a scalar or an array. At theta = 0 the result is
    E_Q[Y^2], the second moment of plain Monte Carlo.
    """
    theta = np.asarray(theta, dtype=float)
    a = theta * sqrt(T)
    b = sigma * sqrt(T)
    d_2 = d2(S0, K, r, sigma, T)
    d_1 = d_2 + b
    discount_squared = exp(-2 * r * T)
    if payoff == "digital":
        return discount_squared * np.exp(a * a) * norm.cdf(d_2 - a)
    if payoff == "call":
        square_bracket = (  # the bracket of eq. (23)
            S0**2 * exp((2 * r + sigma**2) * T) * np.exp(-2 * a * b) * norm.cdf(d_2 + 2 * b - a)
            - 2 * K * S0 * exp(r * T) * np.exp(-a * b) * norm.cdf(d_1 - a)
            + K**2 * norm.cdf(d_2 - a)
        )
        return discount_squared * np.exp(a * a) * square_bracket
    raise ValueError(f"payoff must be 'call' or 'digital', got {payoff!r}")


def theta_optimal(S0, K, r, sigma, T, payoff, bracket=(-5.0, 10.0)):
    """The optimal drift theta*, the unique minimiser of M_2 (Theorem 4.4 of
    the note), computed numerically.

    Since M_2 is strictly convex and coercive, a bounded scalar minimiser on
    ``bracket`` (in units of a = theta sqrt(T)) finds the global minimum, and
    the bracket is only a search window, which is checked afterwards. Why
    minimise log M_2 and not M_2? Because the factor e^{a^2} makes M_2 span
    many orders of magnitude, while its logarithm, which has the same
    minimiser, is much better conditioned.
    """
    def log_M2(a):
        return float(np.log(second_moment(a / sqrt(T), S0, K, r, sigma, T, payoff)))

    optimum = minimize_scalar(log_M2, bounds=bracket, method="bounded", options={"xatol": 1e-10})
    a_star = float(optimum.x)
    if not (bracket[0] + 1e-6 < a_star < bracket[1] - 1e-6):
        raise RuntimeError(f"minimiser a* = {a_star} hit the search bracket {bracket}")
    return a_star / sqrt(T)


def theta_optimal_digital_root(S0, K, r, sigma, T):
    """theta* for the digital as the root of eq. (25) of the note, and not as
    a minimiser.

    It is independent of ``theta_optimal``, a root instead of a minimisation,
    which is what makes it useful to check it. The left side of eq. (25) grows
    linearly and the right side is the Mills ratio, so the equation has
    exactly one root, which is positive (Proposition 4.5).
    """
    d_2 = d2(S0, K, r, sigma, T)

    def mills(x):
        # phi(x) / N(x) in logs: for x << 0 both factors underflow to zero
        # and the plain quotient would be 0/0.
        return float(np.exp(norm.logpdf(x) - norm.logcdf(x)))

    def f(a):
        return 2 * a - mills(d_2 - a)

    # f(0) < 0 and f is increasing. By Proposition 4.5, a* > max(0, -d_2), and
    # since the Mills ratio behaves like a - d_2 for large a, a* < |d_2| + 1
    # amply, so |d_2| + 2 is a safe upper end for the search.
    upper = abs(d_2) + 2.0
    return brentq(f, 1e-12, upper, xtol=1e-12) / sqrt(T)


def _pilot_in_the_money(Z, theta0, S0, K, r, sigma, T, payoff):
    """log(w_theta0 Y^2) and W~_T of the pilot samples in the money, the only
    ones that count in eq. (28) of the note, since the others have Y = 0."""
    if payoff not in ("call", "digital"):
        raise ValueError(f"payoff must be 'call' or 'digital', got {payoff!r}")
    Z = np.asarray(Z, dtype=float)
    W_T = theta0 * T + sqrt(T) * Z  # the pilot's W~_T, drawn under Q^theta0
    h = call_payoff if payoff == "call" else digital_payoff
    Y = exp(-r * T) * h(terminal_price(Z, S0, r, sigma, T, theta0), K)
    in_the_money = Y > 0
    # The weight is read as a function of W~_T, as in the footnote to eq. (28).
    log_w0 = -theta0 * W_T + 0.5 * theta0**2 * T
    return 2 * np.log(Y[in_the_money]) + log_w0[in_the_money], W_T[in_the_money]


def pilot_second_moment(theta, Z, theta0, S0, K, r, sigma, T, payoff):
    """The pilot estimate of M_2(theta), the sample mean of the right side of
    eq. (28) of the note, from the normals ``Z`` of a pilot run under Q^theta0.

    It is unbiased for every theta, and ``theta`` may be a scalar or an array.
    Each term is w_theta0 w_theta Y^2, with both weights read as functions of
    the pilot's W~_T = theta0 T + sqrt(T) Z, and not the w_theta(Z) of eq.
    (16), which is the weight of a sample drawn under Q^theta. With no sample
    in the money the estimate is 0 for every theta.
    """
    theta = np.asarray(theta, dtype=float)
    log_w0_Y2, W = _pilot_in_the_money(Z, theta0, S0, K, r, sigma, T, payoff)
    flat = theta.reshape(-1)
    exponents = log_w0_Y2[:, None] - np.outer(W, flat) + 0.5 * T * flat**2
    M2_hat = np.exp(exponents).sum(axis=0) / np.size(Z)
    return M2_hat.reshape(theta.shape) if theta.ndim else float(M2_hat[0])


def theta_pilot(Z, theta0, S0, K, r, sigma, T, payoff):
    """The drift theta-hat that minimises the pilot estimate of M_2, which
    estimates theta* without its closed form (Subsection 6.5 of the note).

    The normals ``Z`` are read as a pilot run under Q^theta0, and the main run
    has to use fresh draws, independent of these (Remark 3.10). If no sample
    ends in the money, the estimate of M_2 is 0 for every theta and there is
    nothing to minimise, which is why, out of the money, the pilot should run
    under the heuristic drift and not under Q.
    """
    log_w0_Y2, W = _pilot_in_the_money(Z, theta0, S0, K, r, sigma, T, payoff)
    if W.size == 0:
        raise ValueError("no sample of the pilot ends in the money, so its estimate of M_2 "
                         f"is 0 for every theta (theta0 = {theta0}, {np.size(Z)} samples)")

    def slope(theta):
        # The derivative of log M_2-hat divided by T, theta - sum_i pi_i W_i / T,
        # with pi_i proportional to w_theta0 w_theta Y_i^2 as in eq. (29) of the
        # note. Without the factor e^{theta^2 T / 2} of w_theta, which is common
        # to all of them, the pi_i are a softmax, which does not overflow however
        # large the exponents are.
        return theta - np.dot(softmax(log_w0_Y2 - theta * W), W) / T

    # At the root theta T is a weighted mean of the W_i, eq. (29), so it lies
    # between the smallest and the largest of them, and these bracket it. It
    # can only fall on an end through rounding, or when a single sample is in
    # the money, and then the two ends coincide.
    low, high = W.min() / T, W.max() / T
    if slope(low) >= 0:
        return float(low)
    if slope(high) <= 0:
        return float(high)
    return brentq(slope, low, high, xtol=1e-12)
