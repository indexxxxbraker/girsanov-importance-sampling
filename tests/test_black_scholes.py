"""Tests for the closed-form prices of ``black_scholes``.

These prices are the references for every Monte Carlo test of the
repository, so each of them is checked against something that does not share
their code: published values, a computation by hand, the identity that links
the call and the digital, the limit sigma -> 0 and the no-arbitrage bounds.
"""

import math

import numpy as np
import pytest

from girsanov_is.black_scholes import call_price, digital_call_price

# The base case used throughout the repository.
S0, R, SIGMA, T = 100.0, 0.05, 0.2, 1.0
# Strikes as multiples of S0, from at the money (K = S0) to deep out of the
# money (K = 2 S0).
MONEYNESS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]


def test_call_at_the_money_reference_value():
    # The textbook value for S0 = K = 100, r = 5%, sigma = 20% and T = 1.
    assert call_price(S0, 100.0, R, SIGMA, T) == pytest.approx(10.4506, abs=5e-5)


def test_call_hull_example():
    # The worked example of Hull, Options, Futures and Other Derivatives:
    # S0 = 42, K = 40, r = 10%, sigma = 20% and T = 0.5 give c = 4.76.
    assert call_price(42.0, 40.0, 0.10, 0.20, 0.5) == pytest.approx(4.76, abs=5e-3)


def test_digital_at_the_money_by_hand():
    # With S0 = K, d_2 = (r - sigma^2/2) T / (sigma sqrt(T)) = 0.03 / 0.2 = 0.15.
    # N is computed with math.erf, so that the check does not go through
    # scipy, as the formula does.
    N_d2 = 0.5 * (1.0 + math.erf(0.15 / math.sqrt(2.0)))
    expected = math.exp(-R * T) * N_d2
    assert digital_call_price(S0, 100.0, R, SIGMA, T) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("m", MONEYNESS)
def test_digital_is_minus_strike_derivative_of_call(m):
    # d/dK (S_T - K)^+ = -1{S_T > K} almost surely, so D = -dC/dK, and the
    # two formulas must agree although they are coded separately. The central
    # difference has a truncation error of order (h^2/6)|C'''(K)|, which with
    # h = 0.01 stays below 2e-7 in relative terms on this grid.
    K, h = m * S0, 1e-2
    D_fd = -(call_price(S0, K + h, R, SIGMA, T) - call_price(S0, K - h, R, SIGMA, T)) / (2 * h)
    assert D_fd == pytest.approx(digital_call_price(S0, K, R, SIGMA, T), rel=1e-5)


@pytest.mark.parametrize("K", [90.0, 100.0, 150.0])
def test_zero_volatility_limit(K):
    # As sigma -> 0, S_T -> S0 e^{rT} deterministically, so
    # C -> (S0 - K e^{-rT})^+ and D -> e^{-rT} 1{S0 e^{rT} > K}. The strike
    # K = S0 e^{rT} is left out, since there N(d_2) -> 1/2 and the limit of D
    # is not the indicator.
    sigma = 1e-4
    forward = S0 * math.exp(R * T)
    discount = math.exp(-R * T)
    assert call_price(S0, K, R, sigma, T) == pytest.approx(
        max(S0 - K * discount, 0.0), abs=1e-10
    )
    assert digital_call_price(S0, K, R, sigma, T) == pytest.approx(
        discount * float(forward > K), abs=1e-10
    )


def test_no_arbitrage_bounds_and_monotonicity_on_grid():
    K = np.array(MONEYNESS) * S0
    C = call_price(S0, K, R, SIGMA, T)
    D = digital_call_price(S0, K, R, SIGMA, T)
    discount = math.exp(-R * T)
    assert np.all(C >= np.maximum(S0 - K * discount, 0.0))
    assert np.all(C <= S0)
    assert np.all((D > 0.0) & (D < discount))
    # Both prices fall strictly as the strike moves out of the money.
    assert np.all(np.diff(C) < 0.0)
    assert np.all(np.diff(D) < 0.0)


@pytest.mark.parametrize(
    "bad", [{"S0": 0.0}, {"K": -1.0}, {"sigma": 0.0}, {"T": 0.0}]
)
def test_rejects_non_positive_inputs(bad):
    args = {"S0": S0, "K": 100.0, "r": R, "sigma": SIGMA, "T": T} | bad
    with pytest.raises(ValueError):
        call_price(**args)
    with pytest.raises(ValueError):
        digital_call_price(**args)
