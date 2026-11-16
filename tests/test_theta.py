"""Tests for ``theta``, the two choices of the drift (Section 4 of the
note).

The heuristic must put half of the samples in the money (Proposition 4.1).
The closed-form M_2(theta) of Proposition 4.3 must agree with the sample
second moment of w Y under Q^theta within 4 of its standard errors, estimated
from the sample as sqrt((mu_4 - M_2^2) / n), and at theta = 0 it must reduce
to E_Q[Y^2]. M_2 must be convex, and theta_optimal must beat every point of a
grid (Theorem 4.4). For the digital, the root of eq. (25) and the minimiser
of M_2 must give the same theta*, with a* > max(0, -d_2) and
a* / (-d_2) -> 1 deep out of the money (Proposition 4.5), and for the call
theta* must exceed the heuristic out of the money (Section 4.4, case (ii)).
The 95% interval of plain Monte Carlo, IS(theta_h) and IS(theta*) must cover
the price in 460 to 490 of 500 independent runs at K/S0 = 1.5 for both
payoffs, that is, within 3 standard deviations of a Binomial(500, 0.95).
Finally, the pilot run of Subsection 6.5: its estimate (28) of M_2 must be
unbiased for every theta, its minimiser theta-hat must lie between the
smallest and the largest W~_T / T of the samples in the money, a large pilot
must recover theta*, and a pilot with no sample in the money must refuse to
give a drift.
"""

import math

import numpy as np
import pytest
from scipy.stats import norm

from girsanov_is.black_scholes import call_price, d2, digital_call_price
from girsanov_is.estimators import (
    call_payoff,
    digital_payoff,
    is_call,
    is_digital,
    likelihood_ratio,
    mc_call,
    mc_digital,
    standard_normals,
    terminal_price,
)
from girsanov_is.theta import (
    pilot_second_moment,
    second_moment,
    theta_heuristic,
    theta_optimal,
    theta_optimal_digital_root,
    theta_pilot,
)

S0, R, SIGMA = 100.0, 0.05, 0.2
T = 2.0                       # not 1, so that sqrt(T) != T
MONEYNESS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
N = 10**6
SEED = 20261218
# For each payoff: its payoff function, the importance-sampling and the plain
# Monte Carlo estimators, and the closed-form price, in this order.
PAYOFFS = {"call": (call_payoff, is_call, mc_call, call_price),
           "digital": (digital_payoff, is_digital, mc_digital, digital_call_price)}


@pytest.fixture(scope="module")
def Z():
    """The 10^6 standard normals that all the tests of this file share."""
    return standard_normals(N, SEED)


@pytest.mark.parametrize("m", MONEYNESS)
def test_heuristic_puts_half_the_samples_in_the_money(Z, m):
    # The fraction in the money is a binomial proportion with p = 1/2, so its
    # standard error is sqrt(0.25 / n).
    K = m * S0
    theta_h = theta_heuristic(S0, K, R, SIGMA, T)
    assert theta_h * math.sqrt(T) == pytest.approx(-d2(S0, K, R, SIGMA, T))
    fraction_in_the_money = np.mean(terminal_price(Z, S0, R, SIGMA, T, theta_h) > K)
    assert abs(fraction_in_the_money - 0.5) <= 4 * math.sqrt(0.25 / N)


def test_heuristic_sign_follows_d2():
    # Out of the money (d_2 < 0) it pushes up, and near the money (d_2 > 0)
    # it pushes down.
    assert theta_heuristic(S0, 150.0, R, SIGMA, T) > 0
    assert theta_heuristic(S0, 90.0, R, SIGMA, T) < 0


@pytest.mark.parametrize("payoff", ["call", "digital"])
@pytest.mark.parametrize("theta", [0.0, 0.7, 1.4])
@pytest.mark.parametrize("m", [1.0, 1.4, 1.8])
def test_second_moment_matches_sample(Z, payoff, theta, m):
    K = m * S0
    h = PAYOFFS[payoff][0]
    Y = likelihood_ratio(Z, theta, T) * math.exp(-R * T) * h(terminal_price(Z, S0, R, SIGMA, T, theta), K)
    Y2 = Y**2
    m2_sample = Y2.mean()
    SE = math.sqrt(max(np.mean((Y2 - m2_sample) ** 2), 0.0) / N)
    assert abs(m2_sample - second_moment(theta, S0, K, R, SIGMA, T, payoff)) <= 4 * SE


@pytest.mark.parametrize("payoff", ["call", "digital"])
def test_second_moment_at_zero_drift_is_plain_variance_plus_price_squared(payoff):
    # For the digital E_Q[Y^2] = e^{-2rT} N(d_2) exactly, and for both payoffs
    # M_2(0) > V^2, since the variance of plain Monte Carlo is positive.
    K = 1.3 * S0
    V = PAYOFFS[payoff][3](S0, K, R, SIGMA, T)
    if payoff == "digital":
        assert second_moment(0.0, S0, K, R, SIGMA, T, payoff) == pytest.approx(
            math.exp(-2 * R * T) * norm.cdf(d2(S0, K, R, SIGMA, T)), rel=1e-12)
    assert second_moment(0.0, S0, K, R, SIGMA, T, payoff) > V**2


def test_second_moment_rejects_unknown_payoff():
    with pytest.raises(ValueError):
        second_moment(0.0, S0, 100.0, R, SIGMA, T, "put")


@pytest.mark.parametrize("payoff", ["call", "digital"])
@pytest.mark.parametrize("m", [1.0, 1.5, 2.0])
def test_m2_is_convex_and_theta_star_is_its_minimum(payoff, m):
    K = m * S0
    a = np.linspace(-2.0, 6.0, 161)
    theta = a / math.sqrt(T)
    M2 = second_moment(theta, S0, K, R, SIGMA, T, payoff)
    # Midpoint convexity on the grid, with a relative slack of 1e-12 for
    # rounding.
    assert np.all(M2[1:-1] <= 0.5 * (M2[:-2] + M2[2:]) + 1e-12 * M2[1:-1])
    theta_star = theta_optimal(S0, K, R, SIGMA, T, payoff)
    assert second_moment(theta_star, S0, K, R, SIGMA, T, payoff) <= M2.min() * (1 + 1e-9)


@pytest.mark.parametrize("m", MONEYNESS)
def test_digital_theta_star_from_root_equals_minimiser(m):
    K = m * S0
    assert theta_optimal_digital_root(S0, K, R, SIGMA, T) == pytest.approx(
        theta_optimal(S0, K, R, SIGMA, T, "digital"), abs=1e-6)


@pytest.mark.parametrize("m", MONEYNESS)
def test_digital_theta_star_bounds(m):
    K = m * S0
    a_star = theta_optimal(S0, K, R, SIGMA, T, "digital") * math.sqrt(T)
    assert a_star > max(0.0, -d2(S0, K, R, SIGMA, T))


def test_digital_heuristic_is_asymptotically_optimal():
    # a* / (-d_2) -> 1 as d_2 -> -infinity: the ratio decreases towards one
    # along a sequence of strikes that go deep out of the money.
    ratios = []
    for m in [1.5, 2.0, 3.0, 5.0]:
        K = m * S0
        ratios.append(theta_optimal(S0, K, R, SIGMA, T, "digital") / theta_heuristic(S0, K, R, SIGMA, T))
    assert all(ratio > 1 for ratio in ratios)
    assert all(ratios[i] > ratios[i + 1] for i in range(len(ratios) - 1))
    assert ratios[-1] < 1.03


@pytest.mark.parametrize("m", [1.2, 1.5, 2.0])
def test_call_theta_star_exceeds_heuristic_out_of_the_money(m):
    # Section 4.4, case (ii): the call pays more the further S_T ends above
    # K, so the important region lies above the strike.
    K = m * S0
    assert d2(S0, K, R, SIGMA, T) < 0
    assert theta_optimal(S0, K, R, SIGMA, T, "call") > theta_heuristic(S0, K, R, SIGMA, T)


@pytest.mark.parametrize("payoff", ["call", "digital"])
def test_ci_coverage_three_estimators(payoff):
    K = 1.5 * S0
    _, importance, plain, price = PAYOFFS[payoff]
    exact = price(S0, K, R, SIGMA, T)
    theta_h = theta_heuristic(S0, K, R, SIGMA, T)
    theta_star = theta_optimal(S0, K, R, SIGMA, T, payoff)
    hits = {"mc": 0, "heur": 0, "opt": 0}
    for seed in np.random.SeedSequence(SEED).spawn(500):
        Z = standard_normals(10**4, seed)
        for name, run in (("mc", plain(Z, S0, K, R, SIGMA, T)),
                          ("heur", importance(Z, S0, K, R, SIGMA, T, theta_h)),
                          ("opt", importance(Z, S0, K, R, SIGMA, T, theta_star))):
            hits[name] += run.ci[0] <= exact <= run.ci[1]
    # The number of intervals that cover the price is Binomial(500, 0.95),
    # with mean 475 and standard deviation 4.87, and we accept 3 standard
    # deviations, [460, 490].
    for name, covered in hits.items():
        assert 460 <= covered <= 490, f"{payoff}/{name}: {covered} of 500 cover the price"


@pytest.mark.parametrize("payoff", ["call", "digital"])
def test_pilot_second_moment_is_unbiased(payoff):
    # The mean of the pilot estimate over 400 independent pilots of 500
    # samples under Q^theta_h, against the closed form, within 4 standard
    # errors of that mean, at the drifts of plain Monte Carlo, of the
    # heuristic and of the optimum.
    K = 1.5 * S0
    theta_h = theta_heuristic(S0, K, R, SIGMA, T)
    thetas = np.array([0.0, theta_h, theta_optimal(S0, K, R, SIGMA, T, payoff)])
    estimates = np.array([
        pilot_second_moment(thetas, standard_normals(500, seed), theta_h, S0, K, R, SIGMA, T, payoff)
        for seed in np.random.SeedSequence(SEED).spawn(400)])
    SE = estimates.std(axis=0, ddof=1) / math.sqrt(len(estimates))
    error = estimates.mean(axis=0) - second_moment(thetas, S0, K, R, SIGMA, T, payoff)
    assert np.all(np.abs(error) <= 4 * SE)


@pytest.mark.parametrize("payoff", ["call", "digital"])
@pytest.mark.parametrize("m", [1.0, 2.0])
def test_theta_pilot_minimises_the_pilot_estimate(payoff, m):
    K = m * S0
    theta_h = theta_heuristic(S0, K, R, SIGMA, T)
    Z = standard_normals(1000, SEED)
    theta_hat = theta_pilot(Z, theta_h, S0, K, R, SIGMA, T, payoff)
    W_T = theta_h * T + math.sqrt(T) * Z
    in_the_money = terminal_price(Z, S0, R, SIGMA, T, theta_h) > K
    assert W_T[in_the_money].min() / T <= theta_hat <= W_T[in_the_money].max() / T
    grid = theta_hat + np.linspace(-0.5, 0.5, 101)
    M2_hat = pilot_second_moment(grid, Z, theta_h, S0, K, R, SIGMA, T, payoff)
    assert pilot_second_moment(theta_hat, Z, theta_h, S0, K, R, SIGMA, T, payoff) <= M2_hat.min() * (1 + 1e-12)


@pytest.mark.parametrize("payoff", ["call", "digital"])
@pytest.mark.parametrize("m", [1.0, 1.5, 2.0])
def test_large_pilot_recovers_theta_star(Z, payoff, m):
    # The 10^6 normals of the fixture as a pilot under the heuristic drift.
    # The loss 1 - VRF(theta-hat) / VRF(theta*) is quadratic in the error of
    # theta-hat, whose standard deviation is below 10^-3 for a pilot this
    # large, so both bounds leave a wide margin. At K/S0 = 1 the heuristic
    # alone loses most of the reduction for the call.
    K = m * S0
    theta_h = theta_heuristic(S0, K, R, SIGMA, T)
    theta_star = theta_optimal(S0, K, R, SIGMA, T, payoff)
    theta_hat = theta_pilot(Z, theta_h, S0, K, R, SIGMA, T, payoff)
    V = PAYOFFS[payoff][3](S0, K, R, SIGMA, T)
    variance_star = second_moment(theta_star, S0, K, R, SIGMA, T, payoff) - V**2
    variance_hat = second_moment(theta_hat, S0, K, R, SIGMA, T, payoff) - V**2
    assert abs(theta_hat - theta_star) < 0.01
    assert 1 - variance_star / variance_hat < 1e-4


def test_pilot_without_samples_in_the_money():
    # With Z = 0 under Q, S_T = S0 e^{(r - sigma^2/2) T} is far below K = 2 S0,
    # so no sample is in the money, as easily happens to a short pilot under Q
    # deep out of the money (Subsection 6.1). The estimate of M_2 is then 0
    # for every theta, and there is no drift to return.
    Z = np.zeros(10)
    K = 2 * S0
    estimate = pilot_second_moment(np.array([0.0, 1.0, 3.0]), Z, 0.0, S0, K, R, SIGMA, T, "digital")
    assert np.all(estimate == 0.0)
    with pytest.raises(ValueError, match="in the money"):
        theta_pilot(Z, 0.0, S0, K, R, SIGMA, T, "digital")


@pytest.mark.parametrize("theta0", [0.0, 1.0])
def test_pilot_with_one_sample_in_the_money(theta0):
    # With a single sample in the money, M_2-hat(theta) is a multiple of
    # e^{-theta W + theta^2 T / 2}, minimal at theta = W / T, where W is the
    # W~_T of that sample, theta0 T + sqrt(T) Z, and not sqrt(T) Z: the reading
    # of the footnote to eq. (28), which theta0 = 1 tells apart.
    Z = np.array([0.0, 0.0, 5.0])
    K = 2 * S0
    theta_hat = theta_pilot(Z, theta0, S0, K, R, SIGMA, T, "digital")
    assert theta_hat == pytest.approx((theta0 * T + math.sqrt(T) * 5.0) / T)


def test_pilot_rejects_unknown_payoff():
    with pytest.raises(ValueError):
        theta_pilot(np.ones(10), 0.0, S0, 100.0, R, SIGMA, T, "put")
