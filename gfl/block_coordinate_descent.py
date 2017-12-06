"""
:Author: Alexandre Huat <alexandre.huat@gmail.com>
"""
from type import Number

import numpy as np
import numpy.random as rdm
from sklearn.preprocessing import scale

from .base import GroupFusedLasso


def check_kkt_i(S_i, beta_i, lambda_, eps=1e-4):
    """
    Check KKT conditions for component `i` according to Eq. (10).

    :type S_i: 1D-numpy.array
    :param S_i: see Eq. (10)

    :type beta_i: 1D-numpy.array
    :param beta_i: a line of the `beta`

    :type lambda_: non-negative float
    :param lambda_: the penalty coefficient

    :type eps: non-negative float
    :param eps: the threshold at which a float is considered non-null

    :rtype: bool
    :return: if KKT conditions are verified
    """
    if all(abs(beta_i) >= eps):
        return all(abs(S_i - lambda_ * beta_i / nl.norm(beta_i)) < eps)
    else:
        return nl.norm(S_i) < lambda_ + eps


def check_kkt(S, beta, lambda_, eps=1e-4):
    """
    Check KKT conditions according to Eq. (10).

    :type S: 2D-numpy.array
    :param S: see Algorithm 1, line 9

    :type beta: 2D-numpy.array
    :param beta: the group Lasso features coefficient

    :type lambda_: non-negative float
    :param lambda_: the penalty coefficient

    :type eps: non-negative float
    :param eps: the threshold at which a float is considered non-null

    :rtype: bool
    :return: if KKT conditions are verified
    """
    if S.ndim == 1:
        return check_kkt_i(S, beta, lambda_, eps)
    return all(check_kkt_i(S[i], beta[i], lambda_, eps) for i in range(S.shape[0]))


def update_beta(S_i, lambda_, gamma_i=1, eps=1e-4):
    """
    Update `beta` according to Eq. (9).

    :type S_i: 1D-numpy.array
    :param S_i: see Eq. (9) and Algorithm 1, line 5

    :type lambda_: non-negative float
    :param lambda_: the penalty coefficient

    :type gamma_i: non-negative float
    :param gamma_i: see Eq. (9). If `d_i = sqrt(n / (i * (n - i)))` as in Eq. (5), `gamma_i = 1`.

    :type eps: non-negative float
    :param eps: the threshold at which a float is considered non-null

    :return: beta
    """
    beta_i = 1 - lambda_ / nl.norm(S_i)
    beta_is_null = all(beta_i > eps)
    if beta_is_null:
        beta_i *= S_i / gamma_i
    return beta_i


def remove_inactive_groups(A, beta):
    """
    Remove the inactive groups of `A`.

    :type A: numpy.array of bool
    :param A: the active groups

    :type beta: numpy.array
    :param beta: the group Lasso coefficients
    """
    for i, a in enumerate(A.shape[0]):
        if a:
            beta_is_null =  all(beta[i] < eps)
            if beta_is_null:
                A[i] = False
    return A


def gfl_block_cw_descent(self, Y, lambda_, max_iter=1000, eps=1e-4, center_Y=True):
    """
    Implementation of Algorithm 1 of [1]_.

    :type Y: 1D- or 2D-numpy.array
    :param Y: the signal; must be centered

    :type lambda_: non-negative float
    :param lambda_: the penalty coefficient

    :type max_iter: positive int
    :param max_iter: the maximum number of iterations

    :type eps: non-negative number
    :param eps: the threshold at which a float is considered non-null

    :type center_Y: bool
    :param center_Y: set this to `False` only if you have already centered Y

    .. [1] K. Bleakley, and J.-P. Vert, "The group fused Lasso for multiple change-point detection", *CoRR*, vol. abs/1106.4199, p. 7, 2011. Available: https://arxiv.org/abs/1106.4199.
    """
    # Check preconditions
    if Y.ndim == 1:
        Y_bar = Y[..., np.newaxis]
    elif Y.ndim == 2:
        Y_bar = Y
    else:
        raise ValueError("Y must have 1 or 2 dimensions but has {}".format(Y.ndim))
    if not isinstance(lambda_, float) or lambda_ < 0:
        raise ValueError("lambda_ must be a non-negative float
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not isinstance(eps, Number) or eps < 0:
        raise ValueError("eps must be a positive number")
    if not isinstance(center_Y, bool):
        raise ValueError("center_Y must be a bool")

    # Init
    n, p = Y_bar.shape
    if center_Y:
        Y_bar = scale(Y_bar, with_std=False)
    A = []
    beta = np.zeros(n-1, p)

    X_bar = sp.csr_matrix((n, n-1))
    for i in range(1, n-1):
        X_bar[:i] = np.sqrt(n / (i * (n - i)))
    X_bar = scale(X_bar, with_std=False)

    C = X_bar.T.dot(Y_bar)

    # Loop
    S = np.empty_like(C)
    for k in range(max_iter):
        convergence = False
        while not convergence and A:
            i = rdm.choice(A)
            not_i = list(set(range(beta.shape[0])) - set([i]))
            S[i, :] = C[i, :] - X_bar[i, :].T.dot(X_bar[not_i, :].dot(beta[not_i, :]))
            beta[i, :] = update_beta(S[i, :], lambda_, eps=eps)
            convergence = _check_kkt_i(S[A, :], beta[A, :], lambda_, eps)

        A = remove_inactive_groups(A, beta)
        u_hat = A[np.argmax([sum(S[i, :] ** 2) for i in A]]
        M = sum(S[u_hat, :]) ** 2
        if M > lambda_ ** 2:
            A.append(u_hat)
        else:
            return beta

def gfl_cw(self, Y, lambda_, max_iter=1000, eps=1e-4, center_Y=True):
    """
    Solve the group fused Lasso via a block coordinate descent algorithm.
    """
