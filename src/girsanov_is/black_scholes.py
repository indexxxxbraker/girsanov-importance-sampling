"""Closed-form Black-Scholes prices of the European call and of the digital
call, the references against which every Monte Carlo estimate of the
repository is measured (Section 2.2 of the note).

Both are discounted risk-neutral expectations, V = e^{-rT} E_Q[h(S_T)]. The
call is eq. (5) of the note, from Theorem 5.8 and eq. (5.30) of the thesis,
and the digital, which pays 1 if S_T > K, is eq. (6), term (II) of the
decomposition (5.25) of the thesis times the discount factor. All the
functions accept scalars or NumPy arrays and broadcast.
"""

import numpy as np
from scipy.stats import norm


def _validate(S0, K, sigma, T):
    """Raise ValueError unless S0, K, sigma and T are all strictly positive.

    d_2 takes log(S0 / K) and divides by sigma sqrt(T), so a non-positive
    input would not fail by itself: it would only produce a nan or an inf,
    and a NumPy warning that is easy to miss.
    """
    for name, value in (("S0", S0), ("K", K), ("sigma", sigma), ("T", T)):
        if np.any(np.asarray(value) <= 0):
            raise ValueError(f"{name} must be strictly positive, got {value!r}")


def d2(S0, K, r, sigma, T):
    """d_2 of eq. (5.27) of the thesis, so that Q(S_T > K) = N(d_2).

    That is the probability of ending in the money, the quantity that
    importance sampling tries to increase (Section 3 of the note).
    """
    _validate(S0, K, sigma, T)
    return (np.log(S0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def call_price(S0, K, r, sigma, T):
    """Black-Scholes price of the European call, eq. (5) of the note."""
    d_2 = d2(S0, K, r, sigma, T)
    d_1 = d_2 + sigma * np.sqrt(T)
    return S0 * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)


def digital_call_price(S0, K, r, sigma, T):
    """Black-Scholes price of the cash-or-nothing digital call paying 1,
    eq. (6) of the note."""
    return np.exp(-r * T) * norm.cdf(d2(S0, K, r, sigma, T))
