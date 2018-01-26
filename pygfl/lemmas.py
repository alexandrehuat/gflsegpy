# -*- coding: utf-8 -*-

"""
:Author: Alexandre Huat <alexandre.huat@gmail.com>

This module implements computational lemmas in [1], Annexe A.

.. [1] Kevin Bleakley, Jean-Philippe Vert: The group fused Lasso for multiple change-point detection. _CoRR abs/1106.4199_ (2011)
"""

import itertools as itt
import numpy as np
from .utils import hstack, vstack


def d_weights(n):
    """
    See Eq. (5).
    """
    i = np.arange(1, n)
    return np.sqrt(n / (i * (n - i)))


def XbarTR(d, R):
    """
    See Lemma 5.
    """
    r = R.cumsum(axis=0)
    n, p = R.shape
    return hstack(d, p) * (hstack(range(1, n), p) * r[-1, :] / n - r[:-1, :])


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


def XbarTXbarR(d, R, A=None):
    """
    See Lemma 7.
    """
    n, p = d.shape[0] + 1, R.shape[1]
    d_matrix = hstack(d, p)
    A_ = np.array(A) if A is not None else np.arange(n-1)
    R_tilde = d_matrix[A_] * R
    C = (hstack(A_ + 1, p) * R_tilde).sum(axis=0) / n  # S
    T = np.zeros((n - 1, p))
    T[A_] = R_tilde[::-1, :].cumsum(axis=0)[::-1, :]
    C = (vstack(C, n-1) - T).cumsum(axis=0)  # U
    C = d_matrix * C
    return C


def invXbarTXbarR(d, R, A=None):
    """
    See Lemma 8.
    """
    n, p = d.shape[0] + 1, R.shape[1]
    A_ = np.sort(A) if A is not None else np.arange(n-1)
    if A_.size == 1:
        return R / d[A_[0]] ** 2
    d_matrix = hstack(d, p)
    delta = hstack(A_, p)  # Matrix of indices of A for numpy compliance
    # if A_.size >= 3:
    delta = (R[1:, :] / d_matrix[A_[1:]] - R[:-1, :]) / (d_matrix[A_[:-1]] * (delta[1:] - delta[:-1]))
    C = np.empty((A_.size, R.shape[1]))
    C[0, :] = (R[0, :] / (A_[0] + 1) - delta[0]) / d[A_[0]]
    if A_.size > 2:
        C[1:-1, :] = (delta[:-1] - delta[1:]) / d_matrix[A_[1:-1]]
    C[-1, :] = (delta[-1] + R[-1, :] / (n - (A_[-1] + 1))) / d[A_[-1]]
    return C
