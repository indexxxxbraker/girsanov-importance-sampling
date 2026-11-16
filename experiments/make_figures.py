"""The script that produces Figures 1 to 3 of Section 5 of the note, Figure 4
of its Subsection 6.5 and Table 1 of its Appendix A, from the parameters of
``config.py`` and from nothing else.

Run it from the root of the repository with ``python -m experiments.make_figures``,
or with ``--redraw`` at the end to draw the figures again from their CSVs
without running the experiments (see ``redraw``).
For each figure it writes, in ``results/figures/``, a PDF for the note, a PNG
for the README and a CSV with every number that the figure shows, so that a
reader can check a point without running the code. The table goes to
``results/tables/``, as a CSV and as a LaTeX fragment for the note.

- E1 (Figure 1), the variance reduction factor (Definition 2.2) against the
  moneyness K/S0, for antithetic variates and for importance sampling with
  the heuristic drift (eq. (20)) and with the optimal one (Theorem 4.4). The
  points are estimated by eq. (8) on the same draws, and the lines are the
  closed forms, Proposition 2.4 for antithetic variates and Corollary 3.12
  with Proposition 4.3 for importance sampling.
- E2 (Figure 2), the variance reduction factor of importance sampling
  against the drift a = theta sqrt(T), at one strike, that is, a cut of E1
  along the drift. It is estimated on the same draws for every drift and
  compared with its closed form, with a_h and a* marked.
- E3 (Figure 3), the estimate of the digital price and its confidence
  interval against n, deep out of the money, for plain Monte Carlo and for
  importance sampling with the optimal drift.
- E4 (Table 1), the summary table: for each payoff, strike and estimator,
  the estimate with its interval, the VRF, and the relative cost c and the
  efficiency of eq. (9).
- E5 (Figure 4), the optimal drift estimated by a pilot run under the
  heuristic drift (eq. (28)), against the size m of the pilot, at the strike
  of E2: the median and the 10% and 90% quantiles of a-hat over independent
  pilots, with a* and a_h as references. The CSV adds the VRF that each of
  these drifts gives.

Every number the script writes is reproducible except the times of E4, and
with them c and Eff, which depend on the machine and change slightly from one
run to the next.

Five choices are not forced by the note. Why does E2 show the VRF, and not
the variance itself? So that it reads on the same scale as E1, with plain
Monte Carlo at 1 (a = 0) and the best drift at the top, since by Corollary
3.12 the VRF is largest where M_2 is smallest. Why do E3, and each pilot of
E5, take the first n draws of a single vector, and not a new vector for each
n? So that the figure shows one estimate as it gains samples, which is how a
user would see it, and not unrelated runs. And why are the closed forms drawn
as lines on a finer grid than the points? Because the points are estimates
and the line is what they estimate, so the distance between the two is the
noise and nothing else. Why are the times of E4 measured from Z, and not from
the seed? Because the estimators receive Z, so what is timed is exactly the
work in which they differ. Generating Z costs the same for all of them, and
adding it would bring every c closer to 1, so leaving it out does not flatter
importance sampling. Finally, why does E5 give the VRF of the estimated
drifts in closed form, and not measured? Because the main run that follows a
pilot uses fresh draws (Remark 3.10), so given a-hat its variance is exactly
the closed form, and measuring it would only add the noise of a main run to
that of the pilot.
"""

import csv
import logging
import sys
import time
from dataclasses import replace
from functools import partial
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.integrate import quad
from scipy.stats import norm

from experiments import config
from girsanov_is.black_scholes import call_price, d2, digital_call_price
from girsanov_is.estimators import (
    av_call,
    av_digital,
    call_payoff,
    digital_payoff,
    is_call,
    is_digital,
    mc_call,
    mc_digital,
    standard_normals,
    terminal_price,
    vrf,
)
from girsanov_is.theta import second_moment, theta_heuristic, theta_optimal, theta_pilot

RESULTS = Path(__file__).resolve().parent.parent / "results"

PRICE = {"call": call_price, "digital": digital_call_price}
PAYOFF = {"call": call_payoff, "digital": digital_payoff}
PLAIN = {"call": mc_call, "digital": mc_digital}
ANTITHETIC = {"call": av_call, "digital": av_digital}
IMPORTANCE = {"call": is_call, "digital": is_digital}
TITLE = {"call": "European call", "digital": "Digital call"}

# The figures follow the classic look of the papers and books of the field: a
# full box with the ticks inside on its four sides, no grid, thin lines in pure
# colours, hollow markers in the colour of their line, a plain framed legend
# inside the axes, and Arial for the text and the formulas alike. Each
# estimator keeps its colour, line style and marker in every figure, so that it
# is recognised without going back to the legend, and since the line style and
# the marker are enough to tell the series apart, the figures also read in
# black and white. Plain Monte Carlo is the reference, and it is drawn in black.
STYLE = {
    "plain": dict(color="black", linestyle="-", marker="d", label="plain MC"),
    "antithetic": dict(color="#008000", linestyle=":", marker="^", label="antithetic"),
    "heuristic": dict(color="red", linestyle="-.", marker="s", label=r"IS, $\theta_{\mathrm{h}}$"),
    "optimal": dict(color="blue", linestyle="-", marker="o", label=r"IS, $\theta^*$"),
}
REFERENCE_LINE = dict(color="black", linewidth=0.7, linestyle="--")
# The measured points of E2, small and hollow, so that the closed form shows
# through them.
MEASURED = dict(linestyle="none", marker="o", markersize=2.4, markerfacecolor="none",
                markeredgecolor="black", markeredgewidth=0.5)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
    "mathtext.cal": "sans:italic",  # unused, but it spares a warning about a cursive font
    # The fonts of the text have no big radical for \sqrt, so matplotlib takes it
    # from a fallback font, and the radical of STIX Sans matches the text better
    # than the thin one of Computer Modern, the default.
    "mathtext.fallback": "stixsans",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "black",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "lines.linewidth": 0.9,
    "lines.markersize": 4.5,
    "lines.markeredgewidth": 0.8,
    "errorbar.capsize": 2.5,
    "legend.fancybox": False,
    "legend.edgecolor": "black",
    "legend.framealpha": 1.0,
    "legend.fontsize": 8,
    "legend.handlelength": 2.6,
    "legend.labelspacing": 0.35,
    "patch.linewidth": 0.6,  # the frame of the legend, as thin as the box of the axes
    "savefig.bbox": "tight",
    # TrueType fonts in the PDFs, since some publishers reject the Type 3 fonts
    # that matplotlib embeds by default.
    "pdf.fonttype": 42,
})
# matplotlib warns each time it takes the radical from the fallback font, which
# is just what we want it to do.
logging.getLogger("matplotlib.mathtext").setLevel(logging.ERROR)


def antithetic_vrf_closed(S0, K, r, sigma, T, payoff):
    """VRF(AV) = 1 / (1 + rho) in closed form, Proposition 2.4 of the note.

    With sigma_Y^2 from Proposition 4.3 at theta = 0, the only quantity left
    is E_Q[g(Z) g(-Z)], and the product is nonzero only where both members of
    the pair end in the money, that is, for -d_2 < z < d_2. Out of the money
    (d_2 <= 0) that interval is empty and the result is Proposition 2.5. For
    the digital at d_2 = 0 exactly one member of each pair ends in the money,
    so g(Z) + g(-Z) is constant and the VRF is infinite.
    """
    V = PRICE[payoff](S0, K, r, sigma, T)
    var_Y = float(second_moment(0.0, S0, K, r, sigma, T, payoff)) - V**2
    d_2 = float(d2(S0, K, r, sigma, T))
    product = 0.0
    if d_2 > 0:
        h = PAYOFF[payoff]
        discount = np.exp(-r * T)

        def g_times_g_minus(z):
            S_plus = terminal_price(z, S0, r, sigma, T)
            S_minus = terminal_price(-z, S0, r, sigma, T)
            return discount**2 * h(S_plus, K) * h(S_minus, K) * norm.pdf(z)

        product, _ = quad(g_times_g_minus, -d_2, d_2)
    covariance = product - V**2
    return var_Y / (var_Y + covariance)


def importance_vrf_closed(theta, S0, K, r, sigma, T, payoff):
    """VRF(theta) of Corollary 3.12 of the note, from the closed-form M_2 of
    Proposition 4.3."""
    V = PRICE[payoff](S0, K, r, sigma, T)
    M2_plain = second_moment(0.0, S0, K, r, sigma, T, payoff)
    return (M2_plain - V**2) / (second_moment(theta, S0, K, r, sigma, T, payoff) - V**2)


def run_e1(e1, seed):
    """The points of E1: the estimated VRF of each estimator at each strike,
    with its closed form beside it.

    All the estimators of a strike, and all the strikes, share the same n
    draws.
    """
    Z = standard_normals(e1.n, seed)
    S0, r, sigma, T = config.S0, config.R, config.SIGMA, config.T
    rows = []
    for payoff in ("call", "digital"):
        for m in e1.moneyness:
            K = m * S0
            plain = PLAIN[payoff](Z, S0, K, r, sigma, T)
            theta_h = theta_heuristic(S0, K, r, sigma, T)
            theta_star = theta_optimal(S0, K, r, sigma, T, payoff)
            runs = {
                "antithetic": (ANTITHETIC[payoff](Z, S0, K, r, sigma, T),
                               antithetic_vrf_closed(S0, K, r, sigma, T, payoff)),
                "heuristic": (IMPORTANCE[payoff](Z, S0, K, r, sigma, T, theta_h),
                              importance_vrf_closed(theta_h, S0, K, r, sigma, T, payoff)),
                "optimal": (IMPORTANCE[payoff](Z, S0, K, r, sigma, T, theta_star),
                            importance_vrf_closed(theta_star, S0, K, r, sigma, T, payoff)),
            }
            for estimator, (run, closed) in runs.items():
                rows.append(dict(payoff=payoff, moneyness=m, estimator=estimator,
                                 vrf_estimated=vrf(plain, run), vrf_closed=float(closed)))
    return rows


def closed_e1_curves(e1, points=101):
    """The closed-form VRF of each estimator on a fine grid of K/S0, for the
    lines of E1.

    The grid is refined around the strike at which d_2 = 0, where the
    antithetic VRF of the digital is infinite (see ``antithetic_vrf_closed``).
    On a uniform grid the line would show a peak whose height only says how
    close to that strike the nearest point happens to fall.
    """
    S0, r, sigma, T = config.S0, config.R, config.SIGMA, config.T
    low, high = min(e1.moneyness), max(e1.moneyness)
    m_0 = np.exp((r - 0.5 * sigma**2) * T)  # the K/S0 at which d_2 = 0
    delta = np.logspace(-6, -1, 30)
    near = m_0 * np.exp(sigma * sqrt(T) * np.concatenate([-delta, delta]))  # d_2 = +-delta
    grid = np.unique(np.concatenate([np.linspace(low, high, points),
                                     near[(near >= low) & (near <= high)]]))
    curves = {}
    for payoff in ("call", "digital"):
        curves[payoff] = {"antithetic": [], "heuristic": [], "optimal": []}
        for m in grid:
            K = m * S0
            theta_h = theta_heuristic(S0, K, r, sigma, T)
            theta_star = theta_optimal(S0, K, r, sigma, T, payoff)
            curves[payoff]["antithetic"].append(antithetic_vrf_closed(S0, K, r, sigma, T, payoff))
            curves[payoff]["heuristic"].append(
                float(importance_vrf_closed(theta_h, S0, K, r, sigma, T, payoff)))
            curves[payoff]["optimal"].append(
                float(importance_vrf_closed(theta_star, S0, K, r, sigma, T, payoff)))
    return grid, curves


def run_e2(e2, seed):
    """The points of E2: the VRF of importance sampling at each
    a = theta sqrt(T), estimated on the same draws as plain Monte Carlo, with
    its closed form beside it."""
    Z = standard_normals(e2.n, seed)
    S0, r, sigma, T = config.S0, config.R, config.SIGMA, config.T
    K = e2.moneyness * S0
    rows = []
    for payoff in ("call", "digital"):
        plain = PLAIN[payoff](Z, S0, K, r, sigma, T)
        for a in e2.a_grid:
            theta = a / sqrt(T)
            run = IMPORTANCE[payoff](Z, S0, K, r, sigma, T, theta)
            closed = importance_vrf_closed(theta, S0, K, r, sigma, T, payoff)
            rows.append(dict(payoff=payoff, a=a, vrf_estimated=vrf(plain, run),
                             vrf_closed=float(closed)))
    return rows


def run_e3(e3, seed):
    """The points of E3: the estimate and its confidence interval at each n,
    with the exact price, for plain Monte Carlo and for importance sampling
    with theta*.

    The estimate at n uses the first n draws of one vector (see the module
    docstring). Each row also counts the samples that end in the money, and
    gives the probability (1 - p)^n that none does, with p from eq. (15), under
    Q for plain Monte Carlo and under Q^theta* for importance sampling. When
    none does, every payoff is 0, so s_n = 0 and the interval of eq. (4)
    collapses to {0}, which does not contain the price (Subsection 6.1 of the
    note).
    """
    S0, r, sigma, T = config.S0, config.R, config.SIGMA, config.T
    K = e3.moneyness * S0
    V = PRICE[e3.payoff](S0, K, r, sigma, T)
    d_2 = d2(S0, K, r, sigma, T)
    theta_star = theta_optimal(S0, K, r, sigma, T, e3.payoff)
    Z = standard_normals(max(e3.n_grid), seed)
    rows = []
    for n in e3.n_grid:
        runs = (
            ("plain", 0.0, PLAIN[e3.payoff](Z[:n], S0, K, r, sigma, T, config.CONFIDENCE)),
            ("optimal", theta_star, IMPORTANCE[e3.payoff](Z[:n], S0, K, r, sigma, T, theta_star,
                                                          config.CONFIDENCE)),
        )
        for estimator, theta, run in runs:
            S_T = terminal_price(Z[:n], S0, r, sigma, T, theta)
            p = norm.cdf(d_2 + theta * sqrt(T))
            rows.append(dict(n=n, estimator=estimator, estimate=run.estimate,
                             ci_low=run.ci[0], ci_high=run.ci[1], price=V,
                             in_the_money=int(np.count_nonzero(S_T > K)),
                             prob_none_in_the_money=float((1 - p) ** n)))
    return rows


def median_times(estimators, repeats):
    """The median time, in seconds, of ``repeats`` calls of each estimator.

    The estimators take turns, so that a change in the speed of the machine
    during the measurement affects all of them alike. ``run_e4`` calls this
    after the first run of every estimator, so that no time includes the cost
    of a first call.
    """
    times = {name: [] for name in estimators}
    for _ in range(repeats):
        for name, estimate in estimators.items():
            start = time.perf_counter()
            estimate()
            times[name].append(time.perf_counter() - start)
    return {name: float(np.median(seconds)) for name, seconds in times.items()}


def run_e4(e4, seed):
    """The rows of E4: for each payoff, strike and estimator, the estimate with
    its interval, the VRF, the relative cost c and the efficiency of eq. (9).

    All the estimators share the same n draws. The time of an estimator is the
    median of ``e4.timing_repeats`` runs from Z to its result, and c is its
    ratio to the time of plain Monte Carlo at the same strike. theta* is
    computed once per strike, outside the timed runs, since it comes from the
    closed form of M_2 and is the oracle of Section 4, not part of the
    estimator. Plain Monte Carlo is listed with theta = 0, the case without
    drift of importance sampling.
    """
    Z = standard_normals(e4.n, seed)
    S0, r, sigma, T = config.S0, config.R, config.SIGMA, config.T
    rows = []
    for payoff in e4.payoffs:
        for m in e4.moneyness:
            K = m * S0
            V = PRICE[payoff](S0, K, r, sigma, T)
            thetas = {"plain": 0.0,
                      "heuristic": theta_heuristic(S0, K, r, sigma, T),
                      "optimal": theta_optimal(S0, K, r, sigma, T, payoff)}
            args = (Z, S0, K, r, sigma, T)
            level = dict(confidence=config.CONFIDENCE)
            estimators = {
                "plain": partial(PLAIN[payoff], *args, **level),
                "antithetic": partial(ANTITHETIC[payoff], *args, **level),
                "heuristic": partial(IMPORTANCE[payoff], *args, thetas["heuristic"], **level),
                "optimal": partial(IMPORTANCE[payoff], *args, thetas["optimal"], **level),
            }
            runs = {name: estimate() for name, estimate in estimators.items()}
            seconds = median_times(estimators, e4.timing_repeats)
            for name, run in runs.items():
                reduction = vrf(runs["plain"], run)
                c = seconds[name] / seconds["plain"]
                rows.append(dict(payoff=payoff, moneyness=m, estimator=name,
                                 theta=thetas.get(name, ""), price=V, estimate=run.estimate,
                                 half_width=run.half_width, vrf=reduction,
                                 seconds=seconds[name], cost=c, efficiency=reduction / c))
    return rows


def run_e5(e5, seed):
    """The rows of E5: for each payoff and pilot size m, the median and the
    10% and 90% quantiles of a-hat = theta-hat sqrt(T) over the pilots, and
    the closed-form VRF of each of these drifts, of a_h and of a*.

    Every pilot runs under the heuristic drift, as Subsection 6.5 of the note
    advises, and both payoffs use the same pilots. Each row of the matrix of
    draws is one pilot, and its first m draws are the pilot of size m (see the
    module docstring). In the CSV, vrf_at_q10 is the VRF of the drift a_hat_q10,
    and not a quantile of the VRF.
    """
    S0, r, sigma, T = config.S0, config.R, config.SIGMA, config.T
    K = e5.moneyness * S0
    theta_h = theta_heuristic(S0, K, r, sigma, T)
    pilots = standard_normals(e5.pilots * max(e5.m_grid), seed).reshape(e5.pilots, -1)
    rows = []
    for payoff in ("call", "digital"):
        theta_star = theta_optimal(S0, K, r, sigma, T, payoff)
        vrf_of = partial(importance_vrf_closed, S0=S0, K=K, r=r, sigma=sigma, T=T, payoff=payoff)
        for m in e5.m_grid:
            theta_hat = [theta_pilot(pilot[:m], theta_h, S0, K, r, sigma, T, payoff)
                         for pilot in pilots]
            q10, median, q90 = (float(q) for q in np.quantile(theta_hat, [0.1, 0.5, 0.9]))
            rows.append(dict(
                payoff=payoff, moneyness=e5.moneyness, m=m,
                a_heuristic=theta_h * sqrt(T), a_star=theta_star * sqrt(T),
                a_hat_q10=q10 * sqrt(T), a_hat_median=median * sqrt(T), a_hat_q90=q90 * sqrt(T),
                vrf_heuristic=float(vrf_of(theta_h)), vrf_star=float(vrf_of(theta_star)),
                vrf_at_q10=float(vrf_of(q10)), vrf_at_median=float(vrf_of(median)),
                vrf_at_q90=float(vrf_of(q90))))
    return rows


def write_csv(rows, path):
    # The lines end in \n on every system, as .gitattributes asks of the
    # repository, so that git sees no change when the numbers are the same.
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def decimal_places(half_width):
    """The decimal places that leave ``half_width`` with two significant
    digits (none if it is 0, which happens when no sample is in the money)."""
    if half_width <= 0:
        return 0
    return max(0, 1 - int(np.floor(np.log10(half_width))))


def three_figures(x):
    """``x`` with three significant digits."""
    x = float(f"{x:.3g}")  # rounded first, so that 0.9996 is written 1.00 and not 1.000
    return f"{x:.{max(0, 2 - int(np.floor(np.log10(x))))}f}"


def in_units(text, exponent):
    """``text`` as LaTeX math, times 10^exponent unless the exponent is 0."""
    return f"${text}$" if exponent == 0 else rf"${text} \times 10^{{{exponent}}}$"


def write_e4_tex(rows, path):
    """Write E4 as a LaTeX tabular for the note (it needs booktabs), with one
    block of rows per payoff and strike.

    Each estimate is written with its half-width rounded to two significant
    digits and to the same decimal place, as a measurement is written with its
    error, and the exact price with the decimal places of the most precise
    estimate of its block. Prices below 0.01 are written in units of the
    power of ten just below the exact price, the same for the whole block.
    """
    lines = [
        "% Table E4, written by experiments/make_figures.py (see e4_summary.csv).",
        "% The relative costs c, and with them the efficiencies, are times measured",
        "% on the machine that ran the script.",
        r"\begin{tabular}{rllrrrr}",
        r"\toprule",
        rf"$K/S_0$ & $V$ & estimator & estimate $\pm$ {100 * config.CONFIDENCE:g}\% half-width"
        r" & VRF & $c$ & Eff \\",
        r"\midrule",
    ]
    payoffs = list(dict.fromkeys(row["payoff"] for row in rows))
    for payoff in payoffs:
        lines.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{TITLE[payoff]}}}}} \\")
        strikes = list(dict.fromkeys(row["moneyness"] for row in rows if row["payoff"] == payoff))
        for m in strikes:
            block = [row for row in rows if row["payoff"] == payoff and row["moneyness"] == m]
            V = block[0]["price"]
            exponent = 0 if V >= 0.01 else int(np.floor(np.log10(V)))
            unit = 10.0**exponent
            finest = max(decimal_places(row["half_width"] / unit) for row in block)
            for row in block:
                places = decimal_places(row["half_width"] / unit)
                estimate = f"{row['estimate'] / unit:.{places}f}"
                half_width = f"{row['half_width'] / unit:.{places}f}"
                measurement = rf"{estimate} \pm {half_width}"
                if exponent:
                    measurement = f"({measurement})"  # the power of ten multiplies both
                first = row is block[0]
                cells = [
                    f"{m:.1f}" if first else "",
                    in_units(f"{V / unit:.{finest}f}", exponent) if first else "",
                    STYLE[row["estimator"]]["label"],
                    in_units(measurement, exponent),
                    three_figures(row["vrf"]),
                    f"{row['cost']:.2f}",
                    three_figures(row["efficiency"]),
                ]
                lines.append(" & ".join(cells) + r" \\")
            if m != strikes[-1]:
                lines.append(r"\addlinespace")
        if payoff != payoffs[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8", newline="\n")  # see write_csv


def save(fig, folder, name):
    """Write ``fig`` as PDF (for the note) and PNG (for the README).

    The PDF is written without its creation date, so that running the script
    again changes no file unless a number changes.
    """
    fig.savefig(folder / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(folder / f"{name}.png", dpi=200)
    plt.close(fig)


def hollow(estimator, size=4.5):
    """The arguments that draw the points of ``estimator`` as hollow markers in
    its colour, with no line between them."""
    style = STYLE[estimator]
    return dict(linestyle="none", marker=style["marker"], markersize=size,
                markerfacecolor="none", markeredgecolor=style["color"])


def legend_handle(estimator):
    """The legend entry of ``estimator``: its line, with its marker on it."""
    style = STYLE[estimator]
    return Line2D([], [], color=style["color"], linestyle=style["linestyle"],
                  marker=style["marker"], markerfacecolor="none", label=style["label"])


def plot_e1(rows, grid, curves, folder):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9), sharey=True)
    for ax, payoff in zip(axes, ("call", "digital")):
        ax.axhline(1.0, **REFERENCE_LINE)
        for estimator in ("antithetic", "heuristic", "optimal"):
            style = STYLE[estimator]
            ax.plot(grid, curves[payoff][estimator], color=style["color"],
                    linestyle=style["linestyle"])
            points = [row for row in rows
                      if row["payoff"] == payoff and row["estimator"] == estimator]
            ax.plot([row["moneyness"] for row in points], [row["vrf_estimated"] for row in points],
                    **hollow(estimator))
        ax.set_yscale("log")
        ax.set_title(TITLE[payoff])
        ax.set_xlabel("$K/S_0$")
    # The antithetic line of the digital leaves the frame at d_2 = 0, where it
    # is infinite, and the limits are those of the other lines.
    finite = [v for p in curves for e in ("heuristic", "optimal") for v in curves[p][e]]
    axes[0].set_ylim(0.6 * min(finite), 3 * max(finite))
    axes[0].set_ylabel("VRF")
    # One legend serves both panels, in the corner of the left one that the
    # lines leave empty.
    handles = [Line2D([], [], label="plain MC", **REFERENCE_LINE)]
    handles += [legend_handle(e) for e in ("antithetic", "heuristic", "optimal")]
    axes[0].legend(handles=handles, loc="upper left")
    fig.subplots_adjust(wspace=0.08)
    save(fig, folder, "e1_vrf_against_moneyness")


def plot_e2(rows, e2, folder):
    S0, r, sigma, T = config.S0, config.R, config.SIGMA, config.T
    K = e2.moneyness * S0
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9), sharey=True)
    for ax, payoff in zip(axes, ("call", "digital")):
        ax.axhline(1.0, **REFERENCE_LINE)
        points = [row for row in rows if row["payoff"] == payoff]
        a = [row["a"] for row in points]
        ax.plot(a, [row["vrf_closed"] for row in points], color="black")
        ax.plot(a, [row["vrf_estimated"] for row in points], **MEASURED)
        # The two drifts of E1 are marked on the closed form with their markers
        # of E1, so that the points of E1 at this strike are found here.
        for estimator, theta in (("heuristic", theta_heuristic(S0, K, r, sigma, T)),
                                 ("optimal", theta_optimal(S0, K, r, sigma, T, payoff))):
            ax.plot(theta * sqrt(T), importance_vrf_closed(theta, S0, K, r, sigma, T, payoff),
                    **hollow(estimator, size=6.5), markeredgewidth=1.1, zorder=3)
        ax.set_yscale("log")
        ax.set_title(f"{TITLE[payoff]}, $K/S_0 = {e2.moneyness:g}$")
        ax.set_xlabel(r"$a = \theta\sqrt{T}$")
    axes[0].set_ylabel("VRF")
    drift = dict(linestyle="none", markersize=6.5, markerfacecolor="none", markeredgewidth=1.1)
    handles = [
        Line2D([], [], color="black", label="closed form"),
        Line2D([], [], label="measured", **MEASURED),
        Line2D([], [], label="plain MC", **REFERENCE_LINE),
        Line2D([], [], marker=STYLE["heuristic"]["marker"],
               markeredgecolor=STYLE["heuristic"]["color"], label=r"$a_{\mathrm{h}}$", **drift),
        Line2D([], [], marker=STYLE["optimal"]["marker"],
               markeredgecolor=STYLE["optimal"]["color"], label=r"$a^*$", **drift),
    ]
    axes[0].legend(handles=handles, loc="lower center", ncol=2, columnspacing=1.0)
    fig.subplots_adjust(wspace=0.08)
    save(fig, folder, "e2_vrf_against_drift")


def plot_e3(rows, e3, folder):
    # The prices are read in units of the power of ten just below the exact
    # price, which the axis writes at its top, as the caption of the note says.
    V = rows[0]["price"]
    exponent = int(np.floor(np.log10(V)))
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    ax.axhline(V, label="exact price", **REFERENCE_LINE)
    # The two estimators are drawn slightly to either side of each n, so that
    # their intervals do not hide each other. The points at 0 with no bar, on
    # the left, are runs with no sample in the money, and the text of the note
    # explains them, so the figure carries no annotation.
    for estimator, shift in (("plain", 0.94), ("optimal", 1.06)):
        style = STYLE[estimator]
        points = [row for row in rows if row["estimator"] == estimator]
        n = np.array([row["n"] for row in points]) * shift
        estimate = np.array([row["estimate"] for row in points])
        below = estimate - np.array([row["ci_low"] for row in points])
        above = np.array([row["ci_high"] for row in points]) - estimate
        ax.errorbar(n, estimate, yerr=[below, above], linestyle="none", marker=style["marker"],
                    color=style["color"], markerfacecolor="none", elinewidth=0.8,
                    label=style["label"])
    ax.set_xscale("log")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(exponent, exponent), useMathText=True)
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"$\hat V_n$")
    ax.set_title(f"{TITLE[e3.payoff]}, $K/S_0 = {e3.moneyness:g}$")
    ax.legend(loc="upper right")
    save(fig, folder, "e3_estimate_against_n")


def plot_e5(rows, e5, folder):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9), sharey=True)
    for ax, payoff in zip(axes, ("call", "digital")):
        points = [row for row in rows if row["payoff"] == payoff]
        # a* and a_h keep the colour and the line style of their estimators in
        # Figure 1, and the estimates are drawn with bars as in Figure 3.
        for estimator, key in (("optimal", "a_star"), ("heuristic", "a_heuristic")):
            ax.axhline(points[0][key], color=STYLE[estimator]["color"],
                       linestyle=STYLE[estimator]["linestyle"])
        median = np.array([row["a_hat_median"] for row in points])
        below = median - np.array([row["a_hat_q10"] for row in points])
        above = np.array([row["a_hat_q90"] for row in points]) - median
        pilot = ax.errorbar([row["m"] for row in points], median, yerr=[below, above],
                            linestyle="none", marker="o", color="black", markerfacecolor="none",
                            elinewidth=0.8, label=r"$\hat a$: median, 10% to 90%")
        ax.set_xscale("log")
        ax.set_xlabel("$m$")
        ax.set_title(f"{TITLE[payoff]}, $K/S_0 = {e5.moneyness:g}$")
    axes[0].set_ylabel(r"$a = \theta\sqrt{T}$")
    handles = [
        Line2D([], [], color=STYLE["optimal"]["color"], linestyle=STYLE["optimal"]["linestyle"],
               label=r"$a^*$"),
        Line2D([], [], color=STYLE["heuristic"]["color"],
               linestyle=STYLE["heuristic"]["linestyle"], label=r"$a_{\mathrm{h}}$"),
        pilot,
    ]
    axes[0].legend(handles=handles, loc="center right")
    fig.subplots_adjust(wspace=0.08)
    save(fig, folder, "e5_pilot_drift_against_m")


def make_all(folder=RESULTS, e1=None, e2=None, e3=None, e4=None, e5=None):
    """Run E1 to E5 with their seeds from ``config``, and write the figures,
    with their CSVs, to ``folder/figures`` and the table to ``folder/tables``.

    The experiments default to those of ``config``, and the tests pass smaller
    ones (with ``dataclasses.replace``) to run quickly.
    """
    e1, e2, e3 = e1 or config.E1(), e2 or config.E2(), e3 or config.E3()
    e4, e5 = e4 or config.E4(), e5 or config.E5()
    folder = Path(folder)
    figures, tables = folder / "figures", folder / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    seeds = config.seeds()

    rows = run_e1(e1, seeds["E1"])
    grid, curves = closed_e1_curves(e1)
    write_csv(rows, figures / "e1_vrf_against_moneyness.csv")
    plot_e1(rows, grid, curves, figures)

    rows = run_e2(e2, seeds["E2"])
    write_csv(rows, figures / "e2_vrf_against_drift.csv")
    plot_e2(rows, e2, figures)

    rows = run_e3(e3, seeds["E3"])
    write_csv(rows, figures / "e3_estimate_against_n.csv")
    plot_e3(rows, e3, figures)

    rows = run_e4(e4, seeds["E4"])
    write_csv(rows, tables / "e4_summary.csv")
    write_e4_tex(rows, tables / "e4_summary.tex")

    rows = run_e5(e5, seeds["E5"])
    write_csv(rows, figures / "e5_pilot_drift_against_m.csv")
    plot_e5(rows, e5, figures)
    return folder


def read_csv(path):
    """The rows of a CSV written by ``write_csv``, with the numbers back as int
    or float. They are the same numbers, digit for digit, since the CSV writes
    each one in full."""
    def number(text):
        for kind in (int, float):
            try:
                return kind(text)
            except ValueError:
                pass
        return text

    with open(path, newline="", encoding="utf-8") as f:
        return [{key: number(value) for key, value in row.items()} for row in csv.DictReader(f)]


def redraw(folder=RESULTS):
    """Draw the four figures again from their CSVs, without running the
    experiments.

    Why not simply run ``make_all``? Because it would also time E4 again, and
    the note quotes those times. The CSVs keep every number in full, so the
    figures come out exactly as ``make_all`` draws them, which the tests check.
    """
    figures = Path(folder) / "figures"
    rows = read_csv(figures / "e1_vrf_against_moneyness.csv")
    # The lines of E1 span the strikes of its points, so they are read from the CSV.
    e1 = replace(config.E1(), moneyness=tuple(sorted({row["moneyness"] for row in rows})))
    plot_e1(rows, *closed_e1_curves(e1), figures)
    plot_e2(read_csv(figures / "e2_vrf_against_drift.csv"), config.E2(), figures)
    plot_e3(read_csv(figures / "e3_estimate_against_n.csv"), config.E3(), figures)
    plot_e5(read_csv(figures / "e5_pilot_drift_against_m.csv"), config.E5(), figures)
    return Path(folder)


if __name__ == "__main__":
    folder = redraw() if "--redraw" in sys.argv[1:] else make_all()
    for path in sorted(folder.rglob("*.*")):
        print(path.relative_to(folder.parent))
