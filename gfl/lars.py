#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# """
# Author: Joseph Lefevre
# Work in Progress
# """

# import numpy as np
# from .base import GroupFusedLasso
#
#
# class GFLLARS(GroupFusedLasso):
#     def __init__(self):
#         raise NotImplementedError
#
#     def detect(self, X, Y, k, epsilon=1e-9, verbose=False, weights=None):
#         """
#         Detect the breakpoints in X.
#         :param X: a matrix n*p
#         :param Y: a matrix n*p
#         :param k: number of breaking points
#         :return: {"lambda", "jump", "value{i}", "meansignal"} with lambda the estimated lambda values for each change-point, jump the successive change-point positions (1*k), value{i} a i*p matrix of change-point values for the first i change-points, meansignal the mean signal per column (1*p vector)
#         """
#
#         # res is the dict that we will return
#         res = {"lambda": None, "jump": None, "value": [], "meansignal" : None}
#
#         # init of the variables
#         if len(Y.size()) > 1:
#             [n, p] = Y.size()
#         else:
#             n = Y.size()
#             p = 0
#         if weights == None:
#             weights = defaultWeights(n)
#
#         res["meansignal"] = Y.mean()
#         res["lambda"] = np.zeros(k, 1)
#         res["jump"] = np.zeros(k, 1)
#
#         # init of cHat = X'*Y
#         cHat = X.transpose()*Y
#
#         for i in range(0, k, 1):
#             cHatSquareNorm = np.square(cHat).sum(axis=1)
#             bigcHat = np.max(cHatSquareNorm)
#             besti = np.argmax(cHatSquareNorm)
#
#             # In the first iteration, we add the most correlated feature to the active set. For the other iterations, this is already done at the end of the previous iteration
#             if i == 0:
#                 res.jump[0] = besti
#
#             # Compute the descent direction
#             # w = inv(X(:,A)'*X(:,A))*cHat(A,:)
#             A = np.sort(res.jump)
#             I = np.argsort(res.jump)
#
#             w = np.linalg.inv(X[:][A].transpose(), X[:][A]).dot(cHat[A][:])
#             a = X.transpose().dot(X).dot(w)
#
#             # Compute the descent step
#             # For each i we find the largest possible step alpha by solving:
#             # norm(cHat(i,:)-alpha*a(i,:)) = norm(cHat(j,:)-alpha*a(j,:)) where j is in the active set.
#             # We write it as a second order polynomial
#             # a1(i)*alpha^2 - defaultWeights2* a2(i)*alpha + a3(i)
#             a1 = bigcHat - np.square(a).sum(axis=1)
#             a2 = bigcHat - np.multiply(a, cHat).sum(axis=1)
#             a3 = bigcHat - cHatSquareNorm
#
#             # we solve it
#             gammaTemp = np.zeros(2*(n-1), 1)
#
#             # First those where we really have a second-order polynomial
#             subset = np.where(a1 > epsilon)
#             gammeTemp[subset] = np.divide(a2[subset] + np.sqrt(np.square(a2[subset]) - np.multiply(a1[subset], a3[subset])), a1[subset])
#             gammeTemp[subset + n - 1] = np.divide(a2[subset] - np.sqrt(np.square(a2[subset]) - np.multiply(a1[subset], a3[subset])), a1[subset])
#
#             # then those where the quadratic term vanishes and we have a first-order polynomial
#             subset = np.where((a1 <= epsilon) and (a2 > epsilon))
#             gammeTemp[subset] = np.divide(a3[subset], 2*a2[subset])
#             gammeTemp[subset + n - 1] = np.divide(a3[subset], 2*a2[subset])
#
#             # Finally the active set should not be taken into account, as well as
#             # those for which the computation gives dummy solutions
#             maxg = np.max(gammaTemp) + 1
#             subset = np.where(a1 <= epsilon and a2 <= epsilon)
#             gammaTemp[subset] = maxg
#             gammaTemp[subset + n] = maxg
#             gammaTemp[A] = maxg
#             gammaTemp[n + A - 1] = maxg
#             gammaTemp[np.where(gammaTemp <= 0)] = maxg
#             gammaTemp[np.imag(gammaTemp) < 1e-5] = maxg
#
#             # now we can take the minimum
#             gamma = np.min(gammaTemp)
#             nexttoadd = np.argmin(gammaTemp)
#
#             # Update
#             resTemp = np.zeros(i,p)
#             resTemp[I][:] = gamma*w
#             if i > 1:
#                 resTemp[:(i-1)][:] = resTemp[:i-1][:] + res["value"][-1]
#             res["value"].append({i:resTemp})
#             res["lambda"][i] = np.sqrt(bigcHat)
#
#             if i < k:
#                 res["jump"][i+1] = 1 + np.mod(nexttoadd-1, n-1)
#                 cHat = cHat - gamma*a
#     return res
#
# def defaultWeights(n):
#     a = np.array(list(range(1, n))).transpose()
#     w = np.sqrt(np.divide(n,(np.multiply(a,n-a))))
#     return w

# =======================
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

from numbers import Number
import numpy as np
import numpy.random as rdm
import numpy.linalg as npl
from datetime import datetime as dt
from .lemmas import d_weights, XbarTR, invXbarTXbarR, XbarTXbar
from .utils import center_matrix, row_sumsq


def _find_alpha_min(c_hat, a, A, eps=np.finfo(float).eps):
    """
    Equation at line 7 can be reformulated as :math:`||a_{u,\bullet}||^2 ||a_{v,\bullet}||^2 \alpha - 2(\hat{c}_{u,\bullet}^\top a_{u,\bullet} - \hat{c}_{v,\bullet}^\top a_{v,\bullet}) \alpha + ||\hat{c}_{u,\bullet}||^2 ||\hat{c}_{v,\bullet}||^2 = b_2 \alpha^2 + b_1 \alpha + b_0 = 0`.
    """
    # TODO: Finish all

    # Set [1, n-1] \ A
    B = list(set(range(n-1)) - set(A))

    # b coefficients are matrices with u \in B for rows and v \in A for columns
    b_2 = np.outer(row_sumsq(a[B, :]), row_sumsq(a[A, :]))
    b_1 = -2 * (np.outer((a[B, :] * c_hat[B, :]).sum(axis=1), np.ones(len(A))) - np.outer(np.ones(len(B)), (a[A, :] * c_hat[A, :]).sum(axis=1)))
    b_0 = np.outer(row_sumsq(c_hat[B, :]), row_sumsq(c_hat[A, :]))

    alpha = np.full((len(B), len(A)), np.inf)
    # If second-order polynomial
    mask = b_2 > eps
    delta = b_1[mask] ** 2 - 4 * b_2[mask] * b_0[mask]  # Discriminant
    mask &= delta >= 0
    alpha[mask] = (-b_1[mask] - np.sqrt(delta)) / (2 * b_2[mask])
    # Elif first-order polynomial
    mask = ~mask & (b_1 > eps)
    alpha[mask] = b_0[mask] / b_1[mask]

    return alpha.min()


def _gfl_lars(Y_bar, nbpts, eps=np.finfo(float).eps, verbose=1):
    tic = dt.now()
    # Checking parameters
    if Y_bar.ndim != 2:
        raise ValueError("Y must have 2 dimensions but has {}".format(Y.ndim))
    if not isinstance(nbpts, int) and nbpts < 1:
        raise ValueError("nbpts must be a positive int")
    if not isinstance(eps, Number) or eps < 0:
        raise ValueError("eps must be a non-negative number")
    if not isinstance(verbose, int) and verbose < 0:
        raise ValueError("verbose must be a non-negative int")

    # Init
    A = []
    n, p = Y_bar.shape
    d = d_weights(n)
    c_hat = XbarTR(d, Y_bar)

    # Loop
    for k in range(nbpts):
        c_hat_sumsq = (c_hat ** 2).sum(axis=1)  # Instead of norm, sum squares to speed up computations
        if not k:  # First change-point
            u_hat = np.argmin(c_hat_sumsq)
            A.append(u_hat)

        # Descent direction
        w = invXbarTXbarR(d, c_hat[A, :], A)
        a = XbarTXbar(d, w)

        # Descent step: Equation at line 7 can be reformulated as :math:`||a_{u,\bullet}||^2 ||a_{v,\bullet}||^2 \alpha - 2(\hat{c}_{u,\bullet}^\top a_{u,\bullet} - \hat{c}_{v,\bullet}^\top a_{v,\bullet}) \alpha + ||\hat{c}_{u,\bullet}||^2 ||\hat{c}_{v,\bullet}||^2 = b_2 \alpha^2 + b_1 \alpha + b_0 = 0`.
        # c_hat_sumsqmax = c_hat_sumsq.max()
        # b2 = c_hat_sumsqmax - (a ** 2).sum(axis=1)
        # b1 = c_hat_sumsqmax - (a * c_hat).sum(axis=1)
        # b0 = c_hat_sumsqmax - c_hat_sumsq
        # ## Case 1: Second-order polynomial
        # mask = b2 > eps
        alpha_u_hat = _find_alpha_min(c_hat, a, A, eps)

        # Finally
        u_hat = np.argmin(c_hat_sumsq)
        A.append(u_hat)
        c_hat -= alpha_u_hat * a

        # Verbsose
        if verbose >= 1:
            if len(A) < 5:
                verb = "time={}, nbpts={}/{}, active_groups={}".format(dt.now() - tic, k, max_iter, A)
            else:
                verb = "time={}, iter={}/{}, active_groups=[..., {}]"\
                       .format(dt.now() - tic, k, ", ".join(map(str, A[-3:])))
            print(verb, end="\r")

    return [i + 1 for i in A]  # There is an offset (the c_hat matrix has n-1 rows for each jump but Y has n)


def gfl_lars(Y, nbpts, center_Y=True, eps=1e-6, verbose=1):
    """
    Solves the group fused LARS.

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
        raise ValueError("Y must have 1 or 2 dimensions but has {}".format(Y.ndim))
    if not isinstance(nbpts, int) and nbpts < 1:
        raise ValueError("nbpts must be a positive int")
    if not isinstance(eps, Number) or eps < 0:
        raise ValueError("eps must be a non-negative number")
    if not isinstance(center_Y, bool):
        raise ValueError("center_Y must be a bool")
    if not isinstance(verbose, int) and verbose < 0:
        raise ValueError("verbose must be a non-negative int")

    # Preparing data
    if center_Y:
        if verbose >= 1:
            print("Centering Y...", end="\r")
        tic = dt.now()
        Y_bar = center_matrix(Y_bar)
        if verbose >= 1:
            print("Y has been centered.    time={}".format(dt.now() - tic))

    if verbose >= 1:
        print("Performing LARS...")
    bpts = _gfl_lars(Y_bar, nbpts, eps, verbose)
    if verbose >= 1:
        print("Done. breakpoints are", bpts)

    return bpts
