"""Tests for ``experiments/make_figures.py``.

A figure of E1 is only as good as the closed forms drawn beside the
estimates, so the two that the script computes itself are checked against
formulas that do not share its code. E2 and E3 are checked where something is
known exactly: the VRF at a = 0, and the samples in the money behind each
estimate of E3. For E4, plain Monte Carlo must come out as the reference of
the table, and the efficiency as VRF / c. In E5 no drift may give more VRF
than a*, and the pilot estimates must close in on a* as the pilot grows. A
small run then checks that the script writes every file, that running it
twice changes none of them, except for the times of E4, and that drawing the
figures again from their CSVs gives them back byte for byte.
"""

import csv
import math
import re
from dataclasses import replace
from pathlib import Path

import pytest
from scipy.stats import norm

from experiments import config, make_figures
from girsanov_is.black_scholes import call_price, d2
from girsanov_is.theta import second_moment

S0, R, SIGMA, T = 100.0, 0.05, 0.2, 2.0      # T = 2, so that sqrt(T) != T

FILES = [f"{name}.{extension}"
         for name in ("e1_vrf_against_moneyness", "e2_vrf_against_drift",
                      "e3_estimate_against_n", "e5_pilot_drift_against_m")
         for extension in ("csv", "pdf", "png")]
TABLES = ["e4_summary.csv", "e4_summary.tex"]
TIMED = {"seconds", "cost", "efficiency"}     # the columns of E4 that are times
# The experiments cut down to run in seconds.
SMALL = dict(e1=replace(config.E1(), moneyness=(1.0, 1.5), n=10**4),
             e2=replace(config.E2(), a_grid=(0.0, 1.0, 2.0), n=10**4),
             e3=replace(config.E3(), n_grid=(100, 1000)),
             e4=replace(config.E4(), moneyness=(1.0, 2.0), n=10**4, timing_repeats=1),
             e5=replace(config.E5(), pilots=20, m_grid=(32, 100)))


@pytest.mark.parametrize("m", [0.8, 0.9, 1.0, 1.05, 1.2, 1.5, 2.0])
def test_antithetic_closed_form_for_the_digital(m):
    # For the digital the integral that the script computes numerically has a
    # closed form. With D = e^{-rT} and p = N(d_2), sigma_Y^2 = D^2 p (1 - p),
    # and for d_2 > 0 the product g(Z) g(-Z) is D^2 on {-d_2 < Z < d_2}, so
    # Cov = D^2 (2p - 1) - D^2 p^2 and 1 / (1 + rho) = p / (2p - 1). For
    # d_2 < 0 the product vanishes and the VRF is (1 - p) / (1 - 2p), as in
    # Proposition 2.5. The strikes cover both cases.
    K = m * S0
    d_2 = d2(S0, K, R, SIGMA, T)
    p = norm.cdf(d_2)
    expected = p / (2 * p - 1) if d_2 > 0 else (1 - p) / (1 - 2 * p)
    closed = make_figures.antithetic_vrf_closed(S0, K, R, SIGMA, T, "digital")
    assert closed == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize("m", [0.8, 0.9, 1.0, 1.05])
def test_antithetic_closed_form_for_the_call(m):
    # The same integral for the call, at strikes with d_2 > 0. Writing
    # S(z) = S0 e^{mu + b z}, with mu = (r - sigma^2/2) T and b = sigma sqrt(T),
    # we have S(z) S(-z) = S0^2 e^{2 mu}, so on {-d_2 < z < d_2}
    # (S(z) - K)(S(-z) - K) = S0^2 e^{2 mu} + K^2 - K (S(z) + S(-z)), and
    # since mu + b^2/2 = rT, both S(z) and S(-z) integrate against phi over
    # that interval to S0 e^{rT} (N(d_2 - b) - N(-d_2 - b)).
    K = m * S0
    d_2 = d2(S0, K, R, SIGMA, T)
    assert d_2 > 0
    mu, b, D = (R - 0.5 * SIGMA**2) * T, SIGMA * math.sqrt(T), math.exp(-R * T)
    mass = 2 * norm.cdf(d_2) - 1
    S_integral = S0 * math.exp(R * T) * (norm.cdf(d_2 - b) - norm.cdf(-d_2 - b))
    product = D**2 * ((S0**2 * math.exp(2 * mu) + K**2) * mass - 2 * K * S_integral)
    V = call_price(S0, K, R, SIGMA, T)
    var_Y = second_moment(0.0, S0, K, R, SIGMA, T, "call") - V**2
    expected = var_Y / (var_Y + product - V**2)          # Proposition 2.4
    closed = make_figures.antithetic_vrf_closed(S0, K, R, SIGMA, T, "call")
    assert closed == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize("payoff", ["call", "digital"])
def test_importance_closed_form_without_drift_and_with_a_safe_one(payoff):
    # theta = 0 is plain Monte Carlo, so its VRF is one. Out of the money,
    # theta = -d_2 / sqrt(T) is the midpoint of the interval (0, -2 d_2 / sqrt(T)]
    # of Remark 3.6, in which the variance falls (Corollary 3.12), so its VRF
    # is larger than one.
    K = 1.5 * S0
    closed = make_figures.importance_vrf_closed
    theta = -d2(S0, K, R, SIGMA, T) / math.sqrt(T)
    assert closed(0.0, S0, K, R, SIGMA, T, payoff) == pytest.approx(1.0)
    assert closed(theta, S0, K, R, SIGMA, T, payoff) > 1.0


def test_e2_without_drift_is_plain_monte_carlo():
    # At a = 0 importance sampling is plain Monte Carlo bit for bit, so the
    # estimated VRF of E2 is exactly one, and not one up to the noise.
    e2 = replace(config.E2(), a_grid=(-0.5, 0.0, 2.0), n=10**4)
    rows = make_figures.run_e2(e2, config.seeds()["E2"])
    for row in rows:
        if row["a"] == 0.0:
            assert row["vrf_estimated"] == 1.0
            assert row["vrf_closed"] == pytest.approx(1.0)


def test_e3_counts_the_samples_in_the_money():
    # The plain estimate of the digital is e^{-rT} k / n, with k the number of
    # samples in the money (Proposition 6.1 of the note), so k can be read
    # back from it, and the chance of k = 0 is (1 - p)^n with p = N(d_2). With
    # k = 0 every payoff is 0 and the interval collapses to {0}, which is what
    # the figure points out.
    e3 = replace(config.E3(), n_grid=(100, 1000, 10**4))
    rows = make_figures.run_e3(e3, config.seeds()["E3"])
    p = norm.cdf(d2(config.S0, e3.moneyness * config.S0, config.R, config.SIGMA, config.T))
    for row in rows:
        if row["estimator"] == "plain":
            k = row["estimate"] * row["n"] * math.exp(config.R * config.T)
            assert row["in_the_money"] == round(k)
            assert row["prob_none_in_the_money"] == pytest.approx((1 - p) ** row["n"])
        if row["in_the_money"] == 0:
            assert row["ci_low"] == row["ci_high"] == 0.0


def test_e4_has_plain_monte_carlo_as_reference():
    # Plain Monte Carlo is compared with itself, so its VRF, its relative cost
    # and its efficiency are exactly one, and every other efficiency is the
    # VRF over the relative cost, eq. (9). At K/S0 = 1.5 the closed-form VRF
    # of both drifts lies between 14 and 73 (Figure 2 of the note), so even
    # with n = 10^4 their estimates have to be above one, whichever way the
    # noise goes.
    e4 = replace(config.E4(), moneyness=(1.5, 2.0), n=10**4, timing_repeats=2)
    rows = make_figures.run_e4(e4, config.seeds()["E4"])
    assert len(rows) == len(e4.payoffs) * len(e4.moneyness) * 4
    for row in rows:
        assert row["seconds"] > 0
        assert row["efficiency"] == pytest.approx(row["vrf"] / row["cost"])
        if row["estimator"] == "plain":
            assert row["vrf"] == row["cost"] == row["efficiency"] == 1.0
        if row["estimator"] in ("heuristic", "optimal") and row["moneyness"] == 1.5:
            assert row["vrf"] > 1.0


def test_e5_pilot_estimates_close_in_on_the_optimum():
    # a* maximises the VRF (Theorem 4.4), so no drift of a row can do better,
    # up to rounding. From 32 to 1000 samples the spread of a-hat should
    # shrink like 1/sqrt(m), about 5.6 times, and we only ask for a factor of
    # 2. With 1000 samples, the median of 100 pilots must be within 0.01 of a*,
    # more than six of its standard deviations.
    e5 = replace(config.E5(), pilots=100, m_grid=(32, 1000))
    rows = make_figures.run_e5(e5, config.seeds()["E5"])
    for payoff in ("call", "digital"):
        small, large = [row for row in rows if row["payoff"] == payoff]
        for row in (small, large):
            assert row["a_hat_q10"] <= row["a_hat_median"] <= row["a_hat_q90"]
            others = (row["vrf_heuristic"], row["vrf_at_q10"], row["vrf_at_median"], row["vrf_at_q90"])
            assert max(others) <= row["vrf_star"] * (1 + 1e-9)
        assert large["a_hat_q90"] - large["a_hat_q10"] < 0.5 * (small["a_hat_q90"] - small["a_hat_q10"])
        assert abs(large["a_hat_median"] - large["a_star"]) < 0.01


def test_table_numbers_are_rounded_as_measurements():
    # A half-width keeps two significant digits, and the VRF, c and Eff three.
    assert [make_figures.decimal_places(h) for h in (0.029, 0.40, 0.0094, 17.0)] == [3, 2, 4, 0]
    assert make_figures.decimal_places(0.0) == 0
    figures = [make_figures.three_figures(x) for x in (0.6912, 2.004, 0.99996, 72.44, 2523.4)]
    assert figures == ["0.691", "2.00", "1.00", "72.4", "2520"]


def without_times(path):
    """The rows of the CSV of E4 without the columns that are times."""
    with open(path, newline="", encoding="utf-8") as f:
        return [{key: value for key, value in row.items() if key not in TIMED}
                for row in csv.DictReader(f)]


def test_small_run_writes_every_file_and_is_reproducible(tmp_path):
    first = make_figures.make_all(tmp_path / "first", **SMALL)
    second = make_figures.make_all(tmp_path / "second", **SMALL)
    assert sorted(path.name for path in (first / "figures").iterdir()) == sorted(FILES)
    assert sorted(path.name for path in (first / "tables").iterdir()) == sorted(TABLES)
    for name in FILES:
        assert ((first / "figures" / name).read_bytes()
                == (second / "figures" / name).read_bytes()), name
    table = Path("tables") / "e4_summary.csv"
    assert without_times(first / table) == without_times(second / table)
    # Where a block is written in units of a power of ten, the power multiplies
    # the estimate and its half-width, so the two go in parentheses.
    tex = (first / "tables" / "e4_summary.tex").read_text(encoding="utf-8")
    for cell in re.findall(r"\$[^$]*\\pm[^$]*\$", tex):
        assert (r"\times" in cell) == cell.startswith("$(")


def test_redraw_gives_back_the_figures_of_the_run(tmp_path):
    # The figures drawn from the CSVs must be those of the run that wrote the
    # CSVs, byte for byte, so that redrawing never changes a figure by itself.
    folder = make_figures.make_all(tmp_path, **SMALL)
    figures = folder / "figures"
    drawn = {name: (figures / name).read_bytes() for name in FILES if not name.endswith(".csv")}
    make_figures.redraw(folder)
    for name, content in drawn.items():
        assert (figures / name).read_bytes() == content, name
