"""Tests for the plain Monte Carlo estimators of ``estimators`` (Section 2 of
the note).

Every Monte Carlo check is statistical, so its tolerance is a number of
standard errors, and the number is not arbitrary. On the grid of strikes the
estimate must be within 4 standard errors of the closed-form price: with 22
comparisons, a correct implementation fails such a check with probability
about 0.1%, whereas with 3 standard errors it would fail about 6% of the
time by pure chance. In the same spirit, the sample variance of the
at-the-money call is compared with its closed form within a tolerance derived
from the standard error of a sample variance, and not from a fixed
percentage. The file also checks that the same seed gives bit-identical
results, and that the CLT interval covers the price about as often as it
claims.
"""

import math

import numpy as np
import pytest

from girsanov_is.black_scholes import call_price, digital_call_price
from girsanov_is.estimators import (
    MCResult,
    call_payoff,
    digital_payoff,
    mc_call,
    mc_digital,
    mc_estimate,
    standard_normals,
    terminal_price,
)

S0, R, SIGMA, T = 100.0, 0.05, 0.2, 1.0
MONEYNESS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
N = 10**6
SEED = 20261218

# Variance of the discounted at-the-money call payoff in closed form, that is,
# M_2(0) - C^2 with M_2 from eq. (23) of the note. It agrees with the 217
# quoted in the portfolio's variance-reduction note.
ATM_CALL_VARIANCE = 216.66


@pytest.fixture(scope="module")
def Z():
    """The 10^6 standard normals that all the tests of this file share."""
    return standard_normals(N, SEED)


def test_standard_normals_reproducible_and_standard():
    Z1 = standard_normals(10**5, SEED)
    Z2 = standard_normals(10**5, SEED)
    assert np.array_equal(Z1, Z2)
    assert not np.array_equal(Z1, standard_normals(10**5, SEED + 1))
    # With 10^5 draws the standard error of the mean is 1/sqrt(10^5), about
    # 0.0032, and that of the variance sqrt(2/10^5).
    assert abs(Z1.mean()) < 4 * 0.0032
    assert abs(Z1.var() - 1.0) < 4 * math.sqrt(2 / 10**5)


def test_terminal_price_matches_formula_and_lognormal_mean(Z):
    S_T = terminal_price(Z, S0, R, SIGMA, T)
    # One draw against eq. (5.26) of the thesis, written out here with
    # math.exp instead of NumPy's exp, which may differ in the last bit, hence
    # the 1e-14 and not an exact comparison.
    z0 = Z[0]
    expected = S0 * math.exp((R - 0.5 * SIGMA**2) * T + SIGMA * math.sqrt(T) * z0)
    assert S_T[0] == pytest.approx(expected, rel=1e-14)
    # E_Q[S_T] = S0 e^{rT}, since the discounted price is a Q-martingale, and
    # the standard error of the sample mean is sd(S_T) / sqrt(n), with
    # sd(S_T) roughly S0 e^{rT} sigma.
    SE = S0 * math.exp(R * T) * SIGMA / math.sqrt(N)
    assert abs(S_T.mean() - S0 * math.exp(R * T)) < 4 * SE


def test_payoffs():
    S_T = np.array([80.0, 100.0, 120.0])
    assert np.array_equal(call_payoff(S_T, 100.0), [0.0, 0.0, 20.0])
    assert np.array_equal(digital_payoff(S_T, 100.0), [0.0, 0.0, 1.0])


def test_mc_estimate_on_a_known_sample():
    Y = np.array([1.0, 2.0, 3.0, 4.0])
    run = mc_estimate(Y, confidence=0.95)
    assert isinstance(run, MCResult)
    assert run.estimate == 2.5
    assert run.variance == pytest.approx(5 / 3)          # the n - 1 of s_n^2
    assert run.std_error == pytest.approx(math.sqrt(5 / 3 / 4))
    assert run.half_width == pytest.approx(1.959964 * run.std_error, rel=1e-6)
    assert run.ci == (run.estimate - run.half_width, run.estimate + run.half_width)
    assert run.n == 4


def test_mc_estimate_requires_two_samples():
    with pytest.raises(ValueError):
        mc_estimate([1.0])


@pytest.mark.parametrize("m", MONEYNESS)
def test_call_within_four_standard_errors(Z, m):
    K = m * S0
    run = mc_call(Z, S0, K, R, SIGMA, T)
    assert abs(run.estimate - call_price(S0, K, R, SIGMA, T)) <= 4 * run.std_error


@pytest.mark.parametrize("m", MONEYNESS)
def test_digital_within_four_standard_errors(Z, m):
    K = m * S0
    run = mc_digital(Z, S0, K, R, SIGMA, T)
    assert abs(run.estimate - digital_call_price(S0, K, R, SIGMA, T)) <= 4 * run.std_error


@pytest.mark.parametrize("T_other", [0.25, 4.0])
def test_prices_with_maturity_not_one(Z, T_other):
    # With T = 1 we have sqrt(T) = T, and a mix-up between sigma T and
    # sigma sqrt(T) would go unnoticed. Two maturities away from 1 catch it.
    K = 1.2 * S0
    c = mc_call(Z, S0, K, R, SIGMA, T_other)
    d = mc_digital(Z, S0, K, R, SIGMA, T_other)
    assert abs(c.estimate - call_price(S0, K, R, SIGMA, T_other)) <= 4 * c.std_error
    assert abs(d.estimate - digital_call_price(S0, K, R, SIGMA, T_other)) <= 4 * d.std_error


def test_atm_call_variance_matches_closed_form(Z):
    run = mc_call(Z, S0, 100.0, R, SIGMA, T)
    # Var(s_n^2) is about (mu_4 - sigma^4) / n, so estimating mu_4 from the
    # sample gives the standard error of the sample variance, and we accept 4
    # of them.
    Y = np.exp(-R * T) * call_payoff(terminal_price(Z, S0, R, SIGMA, T), 100.0)
    mu4 = np.mean((Y - Y.mean()) ** 4)
    SE_var = math.sqrt((mu4 - run.variance**2) / N)
    assert abs(run.variance - ATM_CALL_VARIANCE) <= 4 * SE_var


def test_same_seed_same_result():
    first_run = mc_call(standard_normals(10**5, SEED), S0, 120.0, R, SIGMA, T)
    second_run = mc_call(standard_normals(10**5, SEED), S0, 120.0, R, SIGMA, T)
    assert first_run == second_run


def test_ci_coverage_digital():
    # 200 independent runs of n = 10^4 at K/S0 = 1.2. The number of intervals
    # that cover the price is Binomial(200, 0.95), with mean 190 and standard
    # deviation about 3.1, and we accept 4 standard deviations, [177, 200].
    K = 1.2 * S0
    exact = digital_call_price(S0, K, R, SIGMA, T)
    seeds = np.random.SeedSequence(SEED).spawn(200)
    hits = 0
    for seed in seeds:
        run = mc_digital(standard_normals(10**4, seed), S0, K, R, SIGMA, T)
        hits += run.ci[0] <= exact <= run.ci[1]
    assert 177 <= hits <= 200
