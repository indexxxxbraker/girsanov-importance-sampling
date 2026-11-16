"""Importance sampling via Girsanov's theorem for out-of-the-money options
under Black-Scholes: the code that goes with the note in ``note/``.

``black_scholes`` has the closed-form prices of the European call and of the
digital call, ``estimators`` the three estimators that the note compares
(plain Monte Carlo, antithetic variates and importance sampling with a
constant drift) and the variance reduction factor, and ``theta`` the two
choices of the drift, the heuristic one and the one that minimises the
variance, with the estimate of the latter from a pilot run.
"""

__version__ = "1.0.0"
