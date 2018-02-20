# -*- coding: utf-8 -*-
"""
This module implements the GFL block coordinate descent.
This algorithm is slow but returns the exact solution of the GFL.

See [1]_, Algorithm 1.

Basic usage
-----------
For a basic usage, call :py:func:`gfl_coord` only. It returns the detected change-points of a signal
given a maximum number of breakpoints and :math:`\lambda` a regularization parameter.
Indeed, the greater :math:`\lambda`, the smaller the number of potential breakpoints.
You will have to find a trade-off between a great :math:`\lambda` value, that will speed up convergence, and a small
:math:`\lambda` value that will allow the finding of all breakpoints.

Advanced usage
--------------
For an advanced usage, you can rely on :py:func:`_gfl_coord` (notice the heading underscore).
It returns the optimal GFL coefficients :math:`\\beta \in \mathbb{R}^{(n-1) \\times p}` of a signal given
:math:`\lambda` a regularization parameter.
Then, you will have to extract the change-points from :math:`\\beta` with a method of your choice.

The function :py:func:`_find_breakpoints` performs such post-processing of :math:`\\beta`.
So, as you guess, :py:func:`gfl_coord` simply calls both of the method mentionned above.

Tuning :math:`\lambda`
----------------------
The key parameter of the block coordinate descent is :math:`\lambda` the regularization coefficient.
Before optimizing :math:`\lambda`, remark that the smaller it is, the greater the number of searched
change-points; i.e. the slower the algorithm converges.
Then, an appropriate optimisation strategy would consist in testing a sequence :math:`(\lambda_l)_{l=1}^L`
in decreasing order and stopping when the validation error becomes unacceptable or does not drop.

Tuning weights
--------------
As well as tuning :math:`\lambda`, you may want to set the optimal weights of each signal position in the model.
This functionality will be added in version 1.1. For now, only the *default* weights can be used, that are:
:math:`\\forall i \in \{1, …, n-1\}, d_i = \sqrt{n \over i (n-i)}`.
But, be advised that if you want to detect a single change-point, Bleakley and Vert proved that the default weights are
the best ones (see Theorem 3). Else, I cannot give any recommendation. Please, refer to [1]_ and your own experience.

See also
--------
gflsegpy.lars
"""

from numbers import Number
import numpy as np
import numpy.random as rdm
from numpy.linalg import norm
from datetime import datetime as dt
from .lemmas import *
from .utils import *


def _check_kkt_i(S_i, beta_i, lambda_, eps=1e-6):
    """
    Checks the KKT conditions for component :math:`i` according to (10).

    Parameters
    ----------
    S_i : numpy.array
        See Eq. (10).
    beta_i : numpy.array
    lambda_ : non-negative float
    eps : non-negative float
        The threshold at which a float must be regarded as null.

    Returns
    -------
    bool
        :py:const:`True` if the :math:`i`-th KKT condition is verified, else :py:const:`False`.
    """
    if norm(beta_i) > eps:
        return (abs(S_i - lambda_ * beta_i / norm(beta_i)) <= eps).all()
    else:
        return norm(S_i) <= lambda_ + eps


def _check_kkt(S, beta, lambda_, eps=1e-6):
    """
    Checks the KKT conditions according to (10).

    Parameters
    ----------
    S : numpy.array
    beta : numpy.array
    lambda_ : non-negative float
    eps : non-negative float
        The threshold at which a float must be regarded as null.

    Returns
    -------
    bool
        :py:const:`True` if the KKT conditions are verified, else :py:const:`False`.
    """
    if S.ndim == 1:
        return _check_kkt_i(S, beta, lambda_, eps)
    for i in range(S.shape[0]):
        if not _check_kkt_i(S[i, :], beta[i, :], lambda_, eps):
            return False
    return True


def _update_beta_i(S_i, lambda_):
    """
    Updates :math:`\\beta` according to Eq. (9).

    *N.B. This computation assumes that the weights are defined w.r.t. (5),
    which gives :math:`\gamma_i = 1` in (9).*

    Parameters
    ----------
    S_i : numpy.array
    lambda_ : non-negative float

    Returns
    -------
    numpy.array
        The :math:`i`-th row of beta
    """
    return (1 - lambda_ / norm(S_i)).clip(0) * S_i


def _compute_u_hat_and_M(S, A):
    """
    See line 10.

    Parameters
    ----------
    S : numpy.array
    A : list

    Returns
    -------
    u_hat : int
    M : float
    """
    # u_hat, M = None, -np.inf
    B = np.array(list(set(range(S.shape[0])) - set(A))).astype(int)
    S_sumsq = col_sumsq(S[B, :])
    i = np.nanargmax(S_sumsq)
    u_hat, M = B[i], S_sumsq[i]
    return u_hat, M


def _block_coordinate_descent(Y_bar, lambda_, max_iter=100, eps=1e-6, verbose=0):
    """
    Implements Algorithm 1.

    Parameters
    ----------
    Y_bar : numpy.array of shape (n, p)
        The centered signal.
    lambda_ : non-negative float
    max_iter : positive int
        The maximum number of iterations.
    eps : non-negative number
        The threshold at which a float must be regarded as null.
    verbose : int
        The verbosity level.

    Returns
    -------
    beta : numpy.array of shape (n-1, p)
        The solution of the GFL.
    KKT : bool
        :py:const:`True` if the KKT conditions are verified, else :py:const:`False`.
    niter : int
        The number of performed iterations.
    """
    # Checking parameters
    try:
        n, p = Y_bar.shape
    except ValueError:
        raise ValueError("Y_bar must have 2 dimensions but has {}".format(Y.ndim))
    if not isinstance(lambda_, Number) or lambda_ < 0:
        raise ValueError("lambda_ must be a non-negative number")
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not isinstance(eps, Number) or eps < 0:
        raise ValueError("eps must be a non-negative number")
    if not isinstance(verbose, int) and verbose < 0:
        raise ValueError("verbose must be a non-negative int")

    # Init
    d = d_weights(n)
    A = []
    beta = np.zeros((n-1, p))
    C = XbarTR(d, Y_bar)
    S = np.empty_like(C)
    lambda2 = lambda_ ** 2
    KKT = False

    # Loop
    if verbose >= 1:
        print("Performing block coordinate descent...")
    tic = dt.now()
    for niter in range(1, max_iter + 1):
        # Coordinate descent
        convergence = False
        A_shuffled = rdm.permutation(A).tolist()
        if verbose >= 1:
            if len(A) < 5:
                verb = "time={}, iter={}/{}, active_groups={}".format(dt.now() - tic, niter, max_iter, A)
            else:
                verb = "time={}, iter={}/{}, active_groups=[..., {}] ({})"\
                       .format(dt.now() - tic, niter, max_iter, ", ".join(map(str, A[-3:])), len(A))
            print(verb, end="\r")
        while not convergence and A_shuffled:
            i = A_shuffled.pop()
            not_i = [j for j in range(beta.shape[0]) if j != i]
            S[i, :] = C[i, :] - XbarTXbar(d, [i], not_i).dot(beta[not_i, :])
            beta[i, :] = _update_beta_i(S[i, :], lambda_)
            convergence = _check_kkt(S[A, :], beta[A, :], lambda_, eps)
        A = [i for i in A if norm(beta[i, :]) > eps]  # Remove inactive groups
        if len(A) >= beta.shape[0]:  # If all points are breakpoints
            break
        # Checking the global KKT
        S = C - XbarTXbar(d).dot(beta)
        u_hat, M = _compute_u_hat_and_M(S, A)
        if M > lambda2:
            A.append(u_hat)
        else:
            KKT = True
            break

    if verbose >= 1:
        print(verb)
        print("Done; KKT={}".format(KKT))

    return beta, KKT, niter


def _gfl_coord(Y, lambda_, max_iter=100, center_Y=True, eps=1e-6, verbose=0):
    """
    Solves the GFL via a block coordinate descent.

    Parameters
    ----------
    Y : numpy.array of shape (n,) or (n, p)
        The signal.
    lambda_ : non-negative float
        The Lasso penalty coefficient.
    max_iter : positive int
        The maximum number of iterations.
    eps : non-negative number
        The threshold at which a float must be regarded as null.
    center_Y : bool
        :py:const:`True` if :math:`Y` has not already been centered (columnwise), else :py:const:`False`.
    verbose : int
        The verbosity level.

    Returns
    -------
    beta : numpy.array of shape (n-1, p)
        The solution of the GFL.
    KKT : bool
        :py:const:`True` if the KKT conditions are verified, else :py:const:`False`.
    niter : int
        The number of performed iterations by the algorithm.
    U : numpy.array of shape (n, p)
        The reconstructed signal.

    See also
    --------
    _block_coordinate_descent()
    """
    # Checking parameters
    if Y.ndim == 1:
        Y_bar = Y.reshape(-1, 1)
    elif Y.ndim == 2:
        Y_bar = Y
    else:
        raise ValueError("Y must have 1 or 2 dimensions but has {}".format(Y.ndim))
    if not isinstance(lambda_, Number) or lambda_ < 0:
        raise ValueError("lambda_ must be a non-negative number")
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not isinstance(eps, Number) or eps < 0:
        raise ValueError("eps must be a non-negative number")
    if not isinstance(center_Y, bool):
        raise ValueError("center_Y must be a bool")
    if not isinstance(verbose, int) and verbose < 0:
        raise ValueError("verbose must be a non-negative int")

    # Preparing data
    if center_Y:
        Y_bar = center_matrix(Y_bar)

    # Performing block coordinate descent
    beta, KKT, niter = _block_coordinate_descent(Y_bar, lambda_, max_iter, eps, verbose)

    # Building U
    n, p = Y_bar.shape
    ## Computing X beta
    U = (hstack(d_weights(n), p) * beta).cumsum(axis=0)
    U = np.concatenate([np.zeros((1, p)), U], axis=0)
    ## Adding gamma
    U += vstack((Y - U).sum(axis=0) / n, n)

    return beta, KKT, niter, U


def _sparse_bpts(bpts, n, min_step):
    sparse_bpts = [bpts.pop(0)]
    while len(sparse_bpts) < n and bpts:
        b0 = bpts.pop(0)
        if all(abs(b0 - b) >= min_step for b in sparse_bpts):
            sparse_bpts.append(b0)
    return sparse_bpts


def _find_breakpoints(beta, n=-1, min_step=1, eps=1e-6, verbose=0):
    """
    Post-processes :math:`\\beta` the solution of the GFL to find the breakpoints of the corresponding signal.
    These are the :py:const:`n` successive :math:`b_k = \\arg \max_i \Vert \\beta_{i, \\bullet} \Vert` for :math:`k=1,
    …,n`.

    Parameters
    ----------
    beta : 2D-numpy.array
        The GFL coefficients.
    n : int
        The maximal number of breakpoints to find. If negative, return all.
    min_step : int
        The minimal step between two breakpoints.
        E.g. if potential breakpoints are 90, 98 and 100 and `min_step=3`,
        the retrieved breakpoints will be 90 and 98.
    eps : non-negative number
        The threshold at which a float must be regarded as null.
    verbose : int
        The verbosity level.

    Returns
    -------
    numpy.array of int
        The breakpoints.
    """
    if not isinstance(beta, np.ndarray):
        raise ValueError("beta must be a numpy.ndarray but is a {}".format(type(beta)))
    if not isinstance(min_step, int):
        raise ValueError("min_step must be an int")
    if not isinstance(eps, Number) or eps < 0:
        raise ValueError("eps must be a non-negative number")
    if n < 0:
        n = beta.shape[0]

    if verbose >= 1:
        print("Post-processing...", end="\r")
    # Find breakpoints
    beta_norm = np.apply_along_axis(norm, 1, beta)
    bpts = [i for i in range(len(beta_norm)) if beta_norm[i] > eps]
    bpts = np.array(bpts)[np.argsort(beta_norm[bpts])][::-1] + 1  # Sorting and correcting the offset

    if bpts.size == 0:
        return bpts

    if min_step <= 1:
        bpts = bpts[:n]
    else:
        bpts = _sparse_bpts(bpts.tolist(), n, min_step)

    if verbose >= 1:
        print("Post-processing: breakpoints={}".format(bpts))

    return np.array(bpts)


def gfl_coord(Y, lambda_, nbpts=-1, min_step=1, max_iter=100, center_Y=True, eps=1e-6, verbose=0):
    """
    Solves the GFL via a block coordinate descent.

    Parameters
    ----------
    Y : 1D- or 2D-numpy.array
        The signal.
    lambda_ : non-negative float
        The Lasso penalty coefficient.
    nbpts : int
        The maximum number of breakpoints to find. If negative, return all.
    min_step : int
        The minimal step between two breakpoints.
        E.g. if potential breakpoints are 90, 98 and 100 and `min_step` is 3,
        retrieved breakpoints will be 90 and 98 in the end.
    max_iter : positive int
        The maximum number of iterations of the block coordinate descent.
    eps : non-negative number
        The threshold at which a float must be regarded as null.
    center_Y : bool
        :py:const:`True` if :math:`Y` has not already been centered (columnwise), else :py:const:`False`.
    verbose : int
        The verbosity level.

    Returns
    -------
    numpy.array of int
        The breakpoints indexes.

    See also
    --------
    _gfl_coord(), _find_breakpoints()
    """
    beta, KKT, niter, U = _gfl_coord(Y, lambda_, max_iter, center_Y, eps, verbose)

    bpts = _find_breakpoints(beta, nbpts, min_step, eps, verbose)

    return bpts
