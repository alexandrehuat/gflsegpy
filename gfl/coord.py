"""
:Author: Alexandre Huat <alexandre.huat@gmail.com>

This module implements the group fused Lasso block coordinate descent.
This algorithm returns the exact optimal solution of the group fused Lasso.
See [1]_, Algorithm 1 for computations and notations.

See also
--------
See module `lars` to use the group fused LARS, which is faster but less accurate.

.. [1] Kevin Bleakley, Jean-Philippe Vert: The group fused Lasso for multiple change-point detection. _CoRR abs/1106.4199_ (2011)
"""

from numbers import Number
import numpy as np
import numpy.random as rdm
import numpy.linalg as npl
from datetime import datetime as dt
from .lemmas import *
from .utils import *

def _check_kkt_i(S_i, beta_i, lambda_, eps=1e-6):
    """
    Checks KKT conditions for component `i` according to (10).

    Parameters
    ----------
    S_i : numpy.array
        See Eq. (10).
    beta_i : numpy.array
    lambda_ : non-negative float
    eps : non-negative float
        The machine definition of zero.

    Returns
    -------
    bool
        `True` if the `i`th KKT condition is verified, `False` else.
    """
    if npl.norm(beta_i) > eps:
        return (abs(S_i - lambda_ * beta_i / npl.norm(beta_i)) <= eps).all()
    else:
        return npl.norm(S_i) <= lambda_ + eps


def _check_kkt(S, beta, lambda_, eps=1e-6):
    """
    Checks KKT conditions according to (10).

    Paremeters
    ----------
    S : numpy.array
    beta : numpy.array
    lambda_ : non-negative float
    eps : non-negative float
        The machine definition of zero.


    Returns
    -------
    bool
        `True` if the KKT conditions are verified, `False` else.
    """
    if S.ndim == 1:
        return _check_kkt_i(S, beta, lambda_, eps)
    for i in range(S.shape[0]):
        if not _check_kkt_i(S[i, :], beta[i, :], lambda_, eps):
            return False
    return True


def _update_beta_i(S_i, lambda_):
    """
    Updates `beta` according to (9).

    _N.B. This computation assumes that the weight :math:`d_i` respects (5), which gives :math:`gamma_i = 1` in (9)._

    Parameters
    ----------
    S_i : numpy.array
    lambda_ : non-negative float

    Returns
    -------
    numpy.array
        The `i`th row of beta
    """
    return (1 - lambda_ / npl.norm(S_i)).clip(0) * S_i


def _compute_u_hat_and_M(S, A):
    """
    See Algorithm 1, line 10.

    Parameters
    ----------
    S : numpy.array
    A : list

    Returns
    -------
    u_hat : int
    M : float
    """
    u_hat, M = None, -np.inf
    for i in set(range(S.shape[0])) - set(A):
        sum2_S_i = S[i, :].T.dot(S[i, :])
        if sum2_S_i > M:
            u_hat = i
            M = sum2_S_i
    return u_hat, M


def _block_coordinate_descent(Y_bar, lambda_, max_iter=1000, eps=1e-6, verbose=0):
    """
    Implements Algorithm 1.

    Parameters
    ----------
    Y_bar : numpy.array of shape (n=n_samples, p=n_features)
        The centered signal.
    lambda_ : non-negative float
    max_iter : positive int
        The maximum number of iterations.
    eps : non-negative number
        The machine definition of zero.
    verbose : int
        The verbosity level.

    Returns
    -------
    beta : numpy.array of shape (n-1, p)
        The solution of the group fused Lasso.
    KKT : bool
        `True` if the KKT conditions are verified, `False` else.
    niter : int
        The number of performed iterations.
    """
    tic = dt.now()
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
    for niter in range(1, max_iter + 1):
        # Block coordinate descent
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
            beta[i, :] = _update_beta(S[i, :], lambda_)
            convergence = _check_kkt(S[A, :], beta[A, :], lambda_, eps)
        A = [i for i in A if npl.norm(beta[i, :]) > eps]  # Remove inactive groups
        # Check global KKT
        S = C - XbarTXbar(d).dot(beta)
        u_hat, M = _compute_u_hat_and_M(S, A)
        if M > lambda2:
            A.append(u_hat)
        else:
            KKT = True
            break

    if verbose >= 1:
        print(verb)
        print("Done. KKT={}".format(KKT))
    return beta, KKT, niter


def gfl_coord(Y, lambda_, max_iter=1000, center_Y=True, eps=1e-6, verbose=0):
    """
    Solves the group fused Lasso via a block coordinate descent algorithm [1]_.
    This algorithm gives an exact solution of the method
    but is generally slower than the LARS algorithm.

    Parameters
    ----------
    Y : numpy.array of shape (n, p)
        The signal.
    lambda_ : non-negative float
        The Lasso penalty coefficient.
    max_iter : positive int
        The maximum number of iterations.
    eps : non-negative number
        The machine definition of zero.
    center_Y : bool
        `True` if `Y` must be centered, `False` else.

    Returns
    -------
    beta : numpy.array of shape (n-1, p)
        The solution of the group fused Lasso.
    KKT : bool
        `True` if the KKT conditions are verified, `False` else.
    niter : int
        The number of performed iterations by the algorithm.
    U : numpy.array of shape (n, p)
        The reconstructed signal.

    See also
    --------
    _block_coordinate_descent()

    .. [1] Kevin Bleakley, Jean-Philippe Vert: The group fused Lasso for multiple change-point detection. _CoRR abs/1106.4199_ (2011)
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
        if verbose >= 1:
            print("Centering Y...", end="\r")
        tic = dt.now()
        Y_bar = center_matrix(Y_bar)
        if verbose >= 1:
            print("Y has been centered.    time={}".format(dt.now() - tic))

    # Performing block coordinate descent
    if verbose >= 1:
        print("Performing block coordinate descent...")
    beta, KKT, niter = _block_coordinate_descent(Y_bar, lambda_, max_iter, eps, verbose)

    # Building U
    if verbose >= 1:
        print("Building U...", end="\r")
    tic = dt.now()
    n, p = Y_bar.shaper
    X = np.zeros((n, n-1))
    d = d_weights(n)
    for i in range(1, n):
        X[i, :i] = d[i-1]
    U = X.dot(beta)
    gamma = np.ones((1, n)).dot(Y - U) / n
    U = np.ones((n, 1)).dot(gamma) + U
    if verbose >= 1:
        print("U has been built.    time={}".format(dt.now() - tic))

    return beta, KKT, niter, U


def _sparse_bpts(bpts, min_step):
    sparse_bpts = [bpts.pop(0)]
    while len(sparse_bpts) < n and bpts:
        b0 = bpts.pop(0)
        if all(abs(b0 - b) >= min_step for b in sparse_bpts):
            sparse_bpts.append(b0)
    return sparse_bpts


def find_breakpoints(beta, n=-1, min_step=1, eps=1e-6):
    """
    Post-processes `beta` the solution of the group fused Lasso to get its breakpoints.
    These are given by getting the maxima of the norms of the rows of `beta`.

    Parameters
    ----------
    beta : numpy.array of shape (n-1, p)
        The group fused Lasso coefficients.
    n : int
        The maximum number of breakpoints to retrieve. If negative, return all.
    min_step : int
        The minimal step between two breakpoints.
        E.g. if potential breakpoints are 90, 98 and 100 and `min_step` is 3,
        retrieved breakpoints will be 90 and 98 in the end.
    eps : non-negative number
        The machine definition of zero.

    Returns
    -------
    list of int
        The breakpoints indexes.
    """
    if not isinstance(beta, np.ndarray):
        raise ValueError("beta must be a numpy.ndarray but is a {}".format(type(beta)))
    if not isinstance(min_step, int):
        raise ValueError("min_step must be an int")
    if not isinstance(eps, Number) or eps < 0:
        raise ValueError("eps must be a non-negative number")
    if n < 0:
        n = beta.shape[0]

    # Find breakpoints
    beta_norm = np.apply_along_axis(npl.norm, 1, beta)
    bpts = [i for i in range(len(beta_norm)) if beta_norm[i] > eps]
    bpts = np.array(bpts)[np.argsort(beta_norm[bpts])][::-1]
    if bpts.size:
        bpts = list(bpts + 1)
    else:
        return []

    if min_step <= 1:
        bpts = bpts[:n]
    else:
        bpts = _sparse_bpts(bpts, min_step)

    return bpts
