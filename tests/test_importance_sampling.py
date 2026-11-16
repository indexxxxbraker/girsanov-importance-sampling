"""Tests for the importance-sampling estimators of ``estimators`` (Section 3
of the note).

The estimate is checked against the closed-form price for several drifts and
strikes, within 4 standard errors as for plain Monte Carlo (Proposition 3.9),
with T = 2 so that a mix-up between theta T and theta sqrt(T) cannot hide.
With theta = 0 the result must be plain Monte Carlo bit for bit, a check
that only means something because ``mc_call`` and ``is_call`` are separate
functions that share ``terminal_price``. The weight must be a density, with
mean one under Q^theta and equal to the ratio of the two Gaussian densities
(Section 3.3), and the fraction of samples in the money must follow
Proposition 3.4. Finally, the variance must fall out of the money for a drift
in (0, -2 d_2 / sqrt(T)] (Remark 3.6 and Corollary 3.12), and it can rise
near the money with a drift that is too large: the method is not automatic.
"""

import math

import numpy as np
import pytest
from scipy.stats import norm

from girsanov_is.black_scholes import call_price, d2, digital_call_price
from girsanov_is.estimators import (
    is_call,
    is_digital,
    likelihood_ratio,
    mc_call,
    mc_digital,
    standard_normals,
    terminal_price,
)

S0, R, SIGMA = 100.0, 0.05, 0.2
T = 2.0                      # not 1, so that sqrt(T) != T
MONEYNESS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
THETAS = [-0.3, 0.5, 1.0, 1.5]
N = 10**6
SEED = 20261218


@pytest.fixture(scope="module")
def Z():
    """The 10^6 standard normals that all the tests of this file share."""
    return standard_normals(N, SEED)


def test_weight_formula_and_zero_drift():
    z = np.array([-1.0, 0.0, 2.0])
    theta = 0.7
    a = theta * math.sqrt(T)
    assert np.allclose(likelihood_ratio(z, theta, T), np.exp(-a * z - 0.5 * a * a))
    assert np.array_equal(likelihood_ratio(z, 0.0, T), np.ones(3))


def test_weight_is_ratio_of_gaussian_densities():
    # w(z) = phi(z + a) / phi(z): the N(0, 1) density at W~_T / sqrt(T) = z + a
    # over the N(a, 1) density at the same point (Section 3.3 of the note).
    z = np.linspace(-3, 3, 13)
    theta, a = 0.9, 0.9 * math.sqrt(T)
    assert np.allclose(likelihood_ratio(z, theta, T), norm.pdf(z + a) / norm.pdf(z))


@pytest.mark.parametrize("theta", THETAS)
def test_weight_has_mean_one(Z, theta):
    # E_{Q^theta}[w] = 1, since w = dQ/dQ^theta. As Var(w) = e^{a^2} - 1, the
    # standard error of the sample mean is sqrt((e^{a^2} - 1) / n).
    w = likelihood_ratio(Z, theta, T)
    a = theta * math.sqrt(T)
    SE = math.sqrt((math.exp(a * a) - 1.0) / N)
    assert abs(w.mean() - 1.0) <= 4 * SE


@pytest.mark.parametrize("theta", [0.0, 1.0])
def test_in_the_money_probability_under_q_theta(Z, theta):
    # Proposition 3.4. The fraction of samples in the money is a binomial
    # proportion, with standard error sqrt(p (1 - p) / n).
    K = 1.5 * S0
    S_T = terminal_price(Z, S0, R, SIGMA, T, theta)
    p_theta = norm.cdf(d2(S0, K, R, SIGMA, T) + theta * math.sqrt(T))
    SE = math.sqrt(p_theta * (1 - p_theta) / N)
    assert abs(np.mean(S_T > K) - p_theta) <= 4 * SE


@pytest.mark.parametrize("theta", THETAS)
@pytest.mark.parametrize("m", MONEYNESS)
def test_is_call_within_four_standard_errors(Z, m, theta):
    K = m * S0
    run = is_call(Z, S0, K, R, SIGMA, T, theta)
    assert abs(run.estimate - call_price(S0, K, R, SIGMA, T)) <= 4 * run.std_error


@pytest.mark.parametrize("theta", THETAS)
@pytest.mark.parametrize("m", MONEYNESS)
def test_is_digital_within_four_standard_errors(Z, m, theta):
    K = m * S0
    run = is_digital(Z, S0, K, R, SIGMA, T, theta)
    assert abs(run.estimate - digital_call_price(S0, K, R, SIGMA, T)) <= 4 * run.std_error


@pytest.mark.parametrize("m", MONEYNESS)
def test_zero_drift_equals_plain_monte_carlo_exactly(Z, m):
    # == between two MCResult compares every field, so this is equality bit
    # for bit, and not up to a tolerance.
    K = m * S0
    assert is_call(Z, S0, K, R, SIGMA, T, 0.0) == mc_call(Z, S0, K, R, SIGMA, T)
    assert is_digital(Z, S0, K, R, SIGMA, T, 0.0) == mc_digital(Z, S0, K, R, SIGMA, T)


@pytest.mark.parametrize("m", [1.4, 1.6, 1.8, 2.0])
def test_variance_drops_out_of_the_money_with_safe_drift(Z, m):
    # Remark 3.6 and Corollary 3.12: with d_2 < 0 and theta in
    # (0, -2 d_2 / sqrt(T)], every path in the money is down-weighted, so the
    # variance falls strictly. We take the midpoint of that interval.
    K = m * S0
    d_2 = d2(S0, K, R, SIGMA, T)
    assert d_2 < 0
    theta = -d_2 / math.sqrt(T)
    for plain, importance in ((mc_call, is_call), (mc_digital, is_digital)):
        var_mc = plain(Z, S0, K, R, SIGMA, T).variance
        var_is = importance(Z, S0, K, R, SIGMA, T, theta).variance
        assert var_is < var_mc


def test_variance_can_increase_with_a_bad_drift(Z):
    # At the money, a large positive drift makes the paths that end just above
    # the strike rare and heavy (Section 3.5 of the note), and importance
    # sampling does worse than plain Monte Carlo.
    K = S0
    theta = 3.0 / math.sqrt(T)
    var_mc = mc_digital(Z, S0, K, R, SIGMA, T).variance
    var_is = is_digital(Z, S0, K, R, SIGMA, T, theta).variance
    assert var_is > var_mc
