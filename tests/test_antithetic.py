"""Tests for the antithetic-variates estimators and for ``vrf`` (Sections 2.3
and 2.4 of the note).

The estimator must use n/2 pairs built from the first half of Z, rejecting an
odd n (Definition 2.3), and its standard error must be computed on the n/2
pair means, which is checked against NumPy directly. It must be unbiased,
within 4 standard errors of the closed form on the grid of strikes as for
the other estimators, with T = 2 so that sqrt(T) != T. Out of the money
(d_2 < 0) a path and its antithetic partner are never both in the money, so
VRF(AV) = sigma_Y^2 / (sigma_Y^2 - V^2) exactly (Proposition 2.5), and the
estimate of eq. (8) must agree with it up to the noise. Near the money
VRF(AV) > 1, since the payoffs are monotone, and it decays towards 1 as the
option moves out of the money. For the at-the-money call it must be close to
the 2.0 of the portfolio's scoreboard, and ``vrf`` must give 1 for an
estimator against itself.
"""

import math

import numpy as np
import pytest

from girsanov_is.black_scholes import call_price, d2, digital_call_price
from girsanov_is.estimators import (
    av_call,
    av_digital,
    call_payoff,
    digital_payoff,
    mc_call,
    mc_digital,
    mc_estimate,
    standard_normals,
    terminal_price,
    vrf,
)
from girsanov_is.theta import second_moment

S0, R, SIGMA = 100.0, 0.05, 0.2
T = 2.0                      # not 1, so that sqrt(T) != T
MONEYNESS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
N = 10**6
SEED = 20261218


@pytest.fixture(scope="module")
def Z():
    """The 10^6 standard normals that all the tests of this file share."""
    return standard_normals(N, SEED)


def test_rejects_odd_n():
    with pytest.raises(ValueError):
        av_call(np.zeros(5), S0, 100.0, R, SIGMA, T)


def test_uses_n_half_pairs_and_pair_means(Z):
    # The pair means rebuilt by hand must give exactly the same result, and
    # their standard error must be the sample standard deviation of the pair
    # means over sqrt(n/2), and not over sqrt(n).
    K = 1.2 * S0
    run = av_call(Z, S0, K, R, SIGMA, T)
    assert run.n == N // 2
    Z_half = Z[: N // 2]
    discount = math.exp(-R * T)
    pair_means = 0.5 * (discount * call_payoff(terminal_price(Z_half, S0, R, SIGMA, T), K)
                        + discount * call_payoff(terminal_price(-Z_half, S0, R, SIGMA, T), K))
    by_hand = mc_estimate(pair_means)
    assert run == by_hand
    assert run.std_error == pytest.approx(pair_means.std(ddof=1) / math.sqrt(N // 2))


@pytest.mark.parametrize("m", MONEYNESS)
def test_av_unbiased_within_four_standard_errors(Z, m):
    K = m * S0
    c = av_call(Z, S0, K, R, SIGMA, T)
    d = av_digital(Z, S0, K, R, SIGMA, T)
    assert abs(c.estimate - call_price(S0, K, R, SIGMA, T)) <= 4 * c.std_error
    assert abs(d.estimate - digital_call_price(S0, K, R, SIGMA, T)) <= 4 * d.std_error


def test_vrf_helper(Z):
    mc_run = mc_call(Z, S0, 120.0, R, SIGMA, T)
    assert vrf(mc_run, mc_run) == 1.0
    av_run = av_call(Z, S0, 120.0, R, SIGMA, T)
    assert vrf(mc_run, av_run) == pytest.approx((mc_run.std_error / av_run.std_error) ** 2)


@pytest.mark.parametrize("payoff", ["call", "digital"])
@pytest.mark.parametrize("m", [1.4, 1.8])
def test_av_vrf_out_of_the_money_matches_proposition(Z, payoff, m):
    # Proposition 2.5: for d_2 < 0, Cov(g(Z), g(-Z)) = -V^2 exactly, so
    # VRF(AV) = sigma_Y^2 / (sigma_Y^2 - V^2), with sigma_Y^2 = M_2(0) - V^2
    # from eqs. (22) and (23).
    K = m * S0
    assert d2(S0, K, R, SIGMA, T) < 0
    price = call_price if payoff == "call" else digital_call_price
    plain, antithetic = (mc_call, av_call) if payoff == "call" else (mc_digital, av_digital)
    V = price(S0, K, R, SIGMA, T)
    var_Y = second_moment(0.0, S0, K, R, SIGMA, T, payoff) - V**2
    exact = var_Y / (var_Y - V**2)
    empirical = vrf(plain(Z, S0, K, R, SIGMA, T), antithetic(Z, S0, K, R, SIGMA, T))
    # Each sample variance has a relative error of about sqrt((kappa - 1) / n),
    # with kappa the kurtosis of the payoff, and out of the money the tails are
    # heavy, so a 10% band is a safe envelope at n = 10^6.
    assert empirical == pytest.approx(exact, rel=0.10)


def test_av_vrf_decays_to_one_out_of_the_money(Z):
    # Out of the money the only gain left is the forced covariance -V^2, and
    # it vanishes as the probability of ending in the money goes to zero
    # (Proposition 2.5 and the paragraph after it).
    vrfs = [vrf(mc_digital(Z, S0, m * S0, R, SIGMA, T), av_digital(Z, S0, m * S0, R, SIGMA, T))
            for m in [1.0, 1.4, 1.8, 2.0]]
    assert all(factor > 1 for factor in vrfs)
    assert vrfs[0] > vrfs[1] > vrfs[2] > vrfs[3]
    assert vrfs[-1] < 1.05


def test_av_vrf_atm_call_consistent_with_portfolio():
    # The portfolio's scoreboard reports a VRF of about 2.0 for the
    # at-the-money call with S0 = K = 100, r = 5%, sigma = 20% and T = 1,
    # with the same convention.
    Z1 = standard_normals(N, SEED + 1)
    vrf_atm = vrf(mc_call(Z1, S0, 100.0, R, SIGMA, 1.0), av_call(Z1, S0, 100.0, R, SIGMA, 1.0))
    assert 1.8 < vrf_atm < 2.2
