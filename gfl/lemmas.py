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
    C = np.empty((n-1, p))
    for i in range(n-1):
        C[i, :] = d[i] * (i * r[n-1, :] / n - r[i, :])
    return C


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
    if A is None: A = list(range(n-1))
    if B is None: B = list(range(n-1))
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


def invXTXR(d, R, A=None):
    if A is None: A = list(range(n-1))
    delta = np.empty(len(A)-1)
    for i in range(len(A)-1)):
        aip1, ai = A[i+1], A[i]
        delta[i] = (R[i+1, :]/d[aip1] - R[i, :]/d[ai]) / (aip1 - ai)
        # TODO
