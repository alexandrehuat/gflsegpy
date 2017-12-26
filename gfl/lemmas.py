"""
:Author: Alexandre Huat <alexandre.huat@gmail.com>

This module implements computational lemmas in [1]_, Annexe A.

.. [1] Kevin Bleakley, Jean-Philippe Vert: The group fused Lasso for multiple change-point detection. _CoRR abs/1106.4199_ (2011)
"""

import itertools as itt
import numpy as np


def d_weights(n):
    """
    See Eq. (5).
    """
    return np.array([np.sqrt(n / (i * (n-i))) for i in range(1, n)])


def XbarTR(d, R):
    """
    See Lemma 5.
    """
    r = R.cumsum(axis=0)
    n, p = R.shape
    return d * (np.arange(1, n) * r[-1, :] / n - r[:-1])


def _minmax(a, b):
    """
    Returns the min and the max of `a` and `b`.
    """
    if a <= b:
        return a, b
    else:
        return b, a


def XbarTXbar(d, A=None, B=None):
    """
    See Lemma 6.
    """
    n = len(d) + 1
    if A is None: A = np.arange(n-1)
    if B is None: B = np.arange(n-1)
    V = np.empty((len(A), len(B)))
    for i, j in itt.product(range(len(A)), range(len(B))):
        a, b = A[i], B[j]
        u, v = _minmax(a, b)
        V[i, j] = d[a] * d[b] * (u * (n - v)) / n
    return V


def XbarTXbarR(d, R):
    """
    See Lemma 7.
    """
    n, p = R.shape; n += 1
    d_matrix = np.outer(d, np.ones(p))
    R_tilde = d_matrix * R
    C = (np.outer(range(1, n), np.ones(p)) * R_tilde).sum(axis=0) / n
    T = R_tilde[::-1, :].cumsum(axis=0)[::-1, :]
    C = (C - T[j, :]).cumsum(axis=0)
    C = d_matrix * C
    return C


def invXbarTXbarR(d, R, A=None):
    """
    See Lemma 8.
    """
    A_ = np.array(A) if A is not None else np.arange(n-1)
    delta = (R[1:, :] / d[A_[1:]] - R[:-1, :] / d[A_[:-1]]) / (A_[1:] - A_[:-1])
    C = np.empty((len(A_), R.shape[1]))
    C[0, :] = (R[0, :] / A_[0] - delta[0]) / d[A_[0]]
    C[1:-1, :] = (delta[1:-2] - delta[2:-1]) / d[A_[1:-1]]
    C[-1, :] = (delta[-1] + R[-1, :] / (n - A_[-1])) / d[A_[-1]]
    return C
