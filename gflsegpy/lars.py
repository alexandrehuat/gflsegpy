# -*- coding: utf-8 -*-
"""
This module implements the GFL LARS.
This algorithm is fast but returns an approximate solution of the group fused Lasso.

Simply call the gflsegpy.gfl_lars() function to use it.
In a nutshell, its inputs are the signal and the number of breakpoints to detect,
and it returns the estimated breakpoints.

See [1]_, Algorithm 2.

See also
--------
gflsegpy.coord
"""

import numpy as np
import warnings as W
from datetime import datetime as dt
from .lemmas import d_weights, XbarTR, invXbarTXbarR, XbarTXbarR
from .utils import center_matrix, col_sumsq, hstack, vstack


def _find_alpha_min(c_hat, a, A, eps=0):
    """
    Returns the minimum :math:`\\alpha` according to line 7, and the corresponding :math:`\hat{u}`.

    For the implementation, the equation at line 7 is reformulated:

    .. math::
       ( \Vert a_{u,\\bullet} \Vert^2 - \Vert a_{v,\\bullet} \Vert^2 ) \\alpha^2
       - 2( a_{u,\\bullet}^\mathrm{T} \hat{c}_{u,\\bullet} - a_{v,\\bullet}^\mathrm{T} \hat{c}_{v,\\bullet} ) \\alpha
       + ( \Vert \hat{c}_{u,\\bullet} \Vert^2 - \Vert \hat{c}_{v,\\bullet} \Vert^2 )
       = \\beta_2 \\alpha^2 - 2 \\beta_1 \\alpha + \\beta_0 = 0

    where :math:`\\beta_0, \\beta_1, \\beta_2 \in \mathbb{R}`.
    """
    n = c_hat.shape[0] + 1
    B = np.array(list(set(range(n-1)) - set(A)))  # Set [1, n-1] \ A

    # beta coefficients for each u in B and v in A are put into matrices where rows represent u and columns v
    beta = np.empty((3, len(B), len(A)))
    beta[2] = hstack(col_sumsq(a[B, :]), len(A)) - vstack(col_sumsq(a[A, :]), len(B))
    beta[1] = (hstack((a[B, :] * c_hat[B, :]).sum(axis=1), len(A)) - vstack((a[A, :] * c_hat[A, :]).sum(axis=1), len(B)))
    beta[0] = hstack(col_sumsq(c_hat[B, :]), len(A)) - vstack(col_sumsq(c_hat[A, :]), len(B))

    # For all u, find the min between all v
    alphaB = np.full((len(B), len(A)), np.inf)
    # If second-order polynomial
    mask = abs(beta[2]) > eps
    # W.simplefilter("ignore", RuntimeWarning, lineno=51)
    delta = -np.ones_like(beta[0])
    delta[mask] = beta[1, mask] ** 2 - beta[2, mask] * beta[0, mask]
    mask &= delta >= 0  # If a solution exists
    delta[mask] = np.sqrt(delta[mask])
    alphaB[mask] = (beta[1, mask] - delta[mask]) / beta[2, mask]  # First solution
    alphaBB = alphaB.copy()
    alphaBB[mask] = (beta[1, mask] + delta[mask]) / beta[2, mask]  # Second solution
    mask &= (alphaBB > eps) & (alphaBB < alphaB)
    alphaB[mask] = alphaBB[mask]
    # Elif first-order polynomial
    mask = (abs(beta[2]) <= eps) & (abs(beta[1]) > eps)
    alphaB[mask] = beta[0, mask] / (2 * beta[1, mask])

    # Removing non-positive solutions
    alphaB[alphaB <= eps] = np.inf

    alphaB = np.nanmin(alphaB, axis=1)

    i = np.nanargmin(col_sumsq(c_hat[B, :]))
    u_hat, alpha = B[i], alphaB[i]
    print(alpha)
    return u_hat, alpha


def _find_alpha_min2(c_hat, a, A, eps=0):
    """
    Returns the minimum :math:`\\alpha` according to line 7, and the corresponding :math:`\hat{u}`.

    This function is based upon the authors implementation which does not exactly matches their paper.
    Indeed, the equation is reformulated:

    .. math::
       (C - (\Vert a_{u,\\bullet} \Vert^2 - \Vert a_{v,\\bullet} \Vert^2 )) \\alpha^2
       - 2(C - (a_{u,\\bullet}^\mathrm{T} \hat{c}_{u,\\bullet} - a_{v,\\bullet}^\mathrm{T} \hat{c}_{v,\\bullet} ))
       \\alpha
       + (C - (\Vert \hat{c}_{u,\\bullet} \Vert^2 - \Vert \hat{c}_{v,\\bullet} \Vert^2))
       = \\beta_2 \\alpha^2 - 2 \\beta_1 \\alpha + \\beta_0 = 0

    where :math:`C = \max_i \Vert \\beta_{i, \\bullet} \Vert^2` and :math:`\\beta_0, \\beta_1, \\beta_2 \in \mathbb{R}`.
    """
    # Init
    chat_sumsq = col_sumsq(c_hat)
    n = c_hat.shape[0] + 1
    alpha = np.zeros((n-1, 2))
    beta = np.full((3, n-1), chat_sumsq.max())
    beta[2] -= col_sumsq(a)
    beta[1] -= (a * c_hat).sum(axis=1)
    beta[0] -= chat_sumsq

    # If second-order polynomial
    ind = abs(beta[2]) > eps
    delta = np.sqrt(beta[1, ind] ** 2 - beta[2, ind] * beta[0, ind])
    alpha[ind, 0] = (beta[1, ind] + delta) / beta[2, ind]
    alpha[ind, 1] = (beta[1, ind] - delta) / beta[2, ind]
    # If first-order polynomial
    ind = (abs(beta[2]) <= eps) & (abs(beta[1]) > eps)
    alpha[ind, :] = hstack(beta[0, ind] / (2 * beta[1, ind]), 2)

    # Correcting alpha
    maxp = alpha.max() + 1
    alpha[(abs(beta[2]) <= eps) & (abs(beta[1]) <= eps)] = maxp
    alpha[A, :] = maxp
    alpha[alpha <= eps] = maxp
    alpha[abs(np.imag(alpha)) <= eps] = maxp

    alpha = alpha.min(axis=1)
    u_hat = np.nanargmin(alpha)
    print(u_hat, alpha[u_hat])
    return u_hat, alpha[u_hat]


def _gfl_lars(Y_bar, nbpts, verbose=1):
    """
    Solves the GFL LARS for a centered signal :math:`\\bar{Y}`.

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
    bpts : numpy.array of int
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

    A =  np.array(A) + 1  # Correcting the offset

    if verbose >= 1:
        print(verb)
        print("Done; breakpoints={}".format(A.tolist()))

    return A


def gfl_lars(Y, nbpts, center_Y=True, verbose=0):
    """
    Solves the GFL LARS.

    Parameters
    ----------
    Y : 1D- or 2D-numpy.array
        The signal.
    nbpts : int
        The number of breakpoints to find.
    center_Y : bool
        :py:const:`True` if :math:`Y` has not already been centered (columnwise), else :py:const:`False`.
    verbose : non-negative int
        The verbosity level.

    Returns
    -------
    bpts : numpy.array of int
        The list of breakpoints, in order of finding.
    """
    W.warn("gfl_lars() may be unstable in gflsegpy 1.0.", UserWarning)

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
    bpts = _gfl_lars(Y_bar, nbpts, verbose)

    return bpts
