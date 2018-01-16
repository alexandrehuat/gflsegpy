# -*- coding: utf-8 -*-

"""
:Author: Alexandre Huat <alexandre.huat@gmail.com>

This module implements the group fused LARS.
This algorithm returns an approximation of the group fused Lasso solution.
See [1]_, Algorithm 2 for computations and notations.

See also
--------
See module `coord` to use the group fused Lasso block coordinate descent, which is slower but more accurate.

.. [1] Kevin Bleakley, Jean-Philippe Vert: The group fused Lasso for multiple change-point detection. _CoRR abs/1106.4199_ (2011)
"""

import numpy as np
import warnings as W
from datetime import datetime as dt
from .lemmas import d_weights, XbarTR, invXbarTXbarR, XbarTXbarR
from .utils import center_matrix, col_sumsq, hstack, vstack


def _find_alpha_min2(c_hat, a, A):
    """
    Returns the minimum `alpha` according to line 7, and the corresponding `u_hat`.

    Notes
    -----
    Equation at line 7 can be reformulated as:
    .. math:: `(\lVert a_{u,\bullet} \rVert^2 - \lVert a_{v,\bullet} \rVert^2) \alpha^2
               - 2(a_{u,\bullet}^\top \hat{c}_{u,\bullet} - a_{v,\bullet}^\top \hat{c}_{v,\bullet}) \alpha
               + (\lVert \hat{c}_{u,\bullet} \rVert^2 - \lVert \hat{c}_{v,\bullet} \rVert^2)
               = \beta_2 \alpha^2 + \beta_1 \alpha + \beta_0 = 0`.
    Which is used in this implementation.
    """
    n = c_hat.shape[0] + 1
    B = np.array(list(set(range(n-1)) - set(A)))  # Set [1, n-1] \ A

    # beta coefficients for each u in B and v in A are put into matrices where rows represent u and columns v
    beta = np.empty((3, len(B), len(A)))
    beta[2] = hstack(col_sumsq(a[B, :]), len(A)) - vstack(col_sumsq(a[A, :]), len(B))
    beta[1] = -2 * (hstack((a[B, :] * c_hat[B, :]).sum(axis=1), len(A)) - vstack((a[A, :] * c_hat[A, :]).sum(axis=1), len(B)))
    beta[0] = hstack(col_sumsq(c_hat[B, :]), len(A)) - vstack(col_sumsq(c_hat[A, :]), len(B))

    # For all u, find the min between all v
    alphaB = np.full((len(B), len(A)), np.inf)
    # If second-order polynomial
    mask = abs(beta[2]) > 0
    delta = -np.empty(alphaB.shape)
    delta[mask] = beta[1, mask] ** 2 - 4 * beta[2, mask] * beta[0, mask]
    mask2 = mask & (delta >= 0)  # A solution exists
    W.simplefilter("ignore", lineno=54)
    delta[mask2] = np.sqrt(delta[mask2])
    alphaB[mask2] = (-beta[1, mask2] - delta[mask2]) / beta[2, mask2]  # First solution
    alphaBB = alphaB.copy()
    alphaBB[mask2] = (-beta[1, mask2] + delta[mask2]) / beta[2, mask2]  # Second solution
    mask2 &= (alphaBB < alphaB) & (alphaBB > 0)
    alphaB[mask2] = alphaBB[mask2]
    alphaB[mask2] /= 2
    # Elif first-order polynomial
    mask = ~mask & (abs(beta[1]) > 0)
    alphaB[mask] = -beta[0, mask] / beta[1, mask]

    # Removing non-positive solutions
    alphaB[alphaB <= 0] = np.inf

    alphaB = alphaB.min(axis=1)

    i = np.argmin(col_sumsq(c_hat[B, :]))
    u_hat, alpha = B[i], alphaB[i]
    return u_hat, alpha


def _find_alpha_min(c_hat, a, A):
    """
    This function is based upon the authors implementation (GFLseg) which does not exactly equal to the paper.
    """
    # Init
    chat_sumsq = col_sumsq(c_hat)
    n = c_hat.shape[0] + 1
    alpha = np.full((n-1, 2), np.inf)
    beta = np.full((3, n-1), chat_sumsq.max())
    beta[2] -= col_sumsq(a)
    beta[1] -= col_sumsq(a * c_hat)
    beta[0] -= col_sumsq(chat_sumsq)

    # If second-order polynomial
    ind = np.where(beta[2] > 0)
    delta = np.empty_like(alpha)
    delta[ind] = np.sqrt(beta[1, ind] ** 2 - beta[2, ind] * beta[3, ind])
    alpha[ind, 0] = (beta[1, ind] + delta[ind]) / beta[0, ind]
    alpha[ind, 1] = (beta[1, ind] - delta[ind]) / beta[0, ind]
    # If first-order polynomial
    ind = np.where((beta[2] <= 0) & (beta[1] > 0))
    alpha[ind, :] = beta[2]


def _gfl_lars(Y_bar, nbpts, verbose=1):
    """
    Solves the group fused LARS for a centered signal `Y_bar`.

    Parameters
    ----------
    Y_bar : numpy.array
        The centered signal, must be 1D or 2D.
    nbpts : int
        The number of breakpoints to find.
    verbose : non-negative int
        The verbosity level.

    Returns
    -------
    bpts : list of int
        The list of breakpoints, in order of finding.
    """
    # Checking parameters
    if Y_bar.ndim != 2:
        raise ValueError("Y_bar must have 2 dimensions but has {}.".format(Y_bar.ndim))
    if not isinstance(nbpts, int) and nbpts < 1:
        raise ValueError("nbpts must be a positive int.")
    if not isinstance(verbose, int) and verbose < 0:
        raise ValueError("verbose must be a non-negative int.")

    # Init
    A = []
    n, p = Y_bar.shape
    d = d_weights(n)
    c_hat = XbarTR(d, Y_bar)

    if verbose >= 1:
        print("Performing LARS...")
    tic = dt.now()

    # First change-point
    u_hat = np.argmin(col_sumsq(c_hat))  # Instead of norm, sum squares to speed up computations
    A.append(u_hat)
    if verbose >= 1:
        verb = "time={}, nbpts={}/{}, active_groups={}".format(dt.now() - tic, len(A), nbpts, A)
    # Loop
    while len(A) < nbpts:
        # Descent direction
        w = invXbarTXbarR(d, c_hat[A, :], A)
        a = XbarTXbarR(d, w, A)
        # Descent step
        u_hat, alpha = _find_alpha_min(c_hat, a, A)
        A.append(u_hat)
        c_hat -= alpha * a

        # Verbose
        if verbose >= 1:
            if len(A) < 5:
                verb = "time={}, nbpts={}/{}, active_groups={}".format(dt.now() - tic, len(A), nbpts, A)
            else:
                verb = "time={}, nbpts={}/{}, active_groups=[..., {}]".format(dt.now() - tic, len(A), nbpts, ", ".join(map(str, A[-3:])))
            print(verb, end="\r")

    A =  [i + 1 for i in A]  # There is an offset (the c_hat matrix has n-1 rows for each jump but Y has n)

    if verbose >= 1:
        print(verb)
        print("Done. breakpoints={}".format(A))

    return A


def gfl_lars(Y, nbpts, center_Y=True, verbose=0):
    """
    Solves the group fused LARS.

    Parameters
    ----------
    Y : numpy.array
        The signal, must be 1D or 2D.
    nbpts : int
        The number of breakpoints to find.
    center_Y : bool (default: True)
        If `True`, `Y` will be centered in the function. Set this to `False` if `Y` is already centered.
    verbose : non-negative int
        The verbosity level.

    Returns
    -------
    bpts : list of int
        The list of breakpoints, in order of finding.
    """
    # Checking parameters
    if Y.ndim == 1:
        Y_bar = Y.reshape(-1, 1)
    elif Y.ndim == 2:
        Y_bar = Y
    else:
        raise ValueError("Y must have 1 or 2 dimensions but has {}.".format(Y.ndim))
    if not isinstance(nbpts, int) and nbpts < 1:
        raise ValueError("nbpts must be a positive int.")
    if not isinstance(center_Y, bool):
        raise ValueError("center_Y must be a bool.")
    if not isinstance(verbose, int) and verbose < 0:
        raise ValueError("verbose must be a non-negative int.")

    # Performing LARS
    if center_Y:
        Y_bar = center_matrix(Y_bar)
        Y_bar /= vstack((Y_bar ** 2).sum(axis=0), Y_bar.shape[0])
    bpts = _gfl_lars(Y_bar, nbpts, verbose)

    return bpts
