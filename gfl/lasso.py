"""
:Author: Alexandre Huat <alexandre.huat@gmail.com>
"""
from numbers import Number
import numpy as np
import numpy.random as rdm
import numpy.linalg as npl
from scipy.sparse import csr_matrix
from sklearn.preprocessing import scale
from datetime import datetime as dt
import matplotlib.pyplot as plt


def check_kkt_i(S_i, beta_i, lambda_, eps=1e-4):
    """
    Checks KKT conditions for component `i` according to Eq. (10) [1]_.

    Parameters
    ----------
    S_i : numpy.array
        See Eq. (10).
    beta_i : numpy.array
        The `i`th Lasso coefficients.
    lambda_ : non-negative float
        The Lasso penalty coefficient
    eps : non-negative float
        The threshold at which a float is considered non-null.

    Returns
    -------
    bool
        `True` if the KKT condition is verified, `False` else.

    .. [1] Jean-Philippe Vert, Kevin Bleakley: The group fused Lasso for multiple change-point detection. _CoRR abs/1106.4199_ (2011)
    """
    if npl.norm(beta_i) >= eps:
        return (abs(S_i - lambda_ * beta_i / npl.norm(beta_i)) < eps).all()
    else:
        return npl.norm(S_i) < lambda_ + eps


def check_kkt(S, beta, lambda_, eps=1e-4):
    """
    Checks KKT conditions according to Eq. (10) [1]_.

    Paremeters
    ----------
    S : numpy.array
        See Algorithm 1, line 9.
    beta : numpy.array
        The Lasso coefficients.
    lambda_ : non-negative float
        The Lasso penalty coefficient
    eps : non-negative float
        The threshold at which a float is considered non-null.

    Returns
    -------
    bool
        `True` if the KKT conditions are verified, `False` else.

    .. [1] Jean-Philippe Vert, Kevin Bleakley: The group fused Lasso for multiple change-point detection. _CoRR abs/1106.4199_ (2011)
    """
    if S.ndim == 1:
        return check_kkt_i(S, beta, lambda_, eps)
    for i in range(S.shape[0]):
        if not check_kkt_i(S[i, :], beta[i, :], lambda_, eps):
            return False
    return True


def update_beta(S_i, lambda_, gamma_i):
    """
    Updates `beta` according to Eq. (9) [1]_.

    Parameters
    ----------
    S_i : numpy.array
        See Eq. (9) and Algorithm 1, line 5.
    lambda_ : non-negative float
        The Lasso penalty coefficient.
    gamma_i : non-negative float
        See Eq. (9).

    Returns
    -------
    numpy.array
        The `i`th row of beta

    .. [1] Jean-Philippe Vert, Kevin Bleakley: The group fused Lasso for multiple change-point detection. _CoRR abs/1106.4199_ (2011)
    """
    return (1 - lambda_ / npl.norm(S_i)).clip(0) * S_i / gamma_i


def compute_u_hat_and_M(S, A):
    """
    Finds `u_hat` and compute `M` according to Algorithm 1, line 10 [1]_.

    Parameters
    ----------
    S : numpy.array
        See Algorithm 1, line 9.
    A : list
        The active groups of `beta`.

    Returns
    -------
    u_hat : int
        See Algorithm 1, line 10.
    M : float
        See Algorithm 1, line 10.

    .. [1] Jean-Philippe Vert, Kevin Bleakley: The group fused Lasso for multiple change-point detection. _CoRR abs/1106.4199_ (2011)
    """
    M = -np.inf
    for i in set(range(S.shape[0])) - set(A):
        # print(S[i, :].max())
        sum2_S_i = S[i, :].T.dot(S[i, :])
        if sum2_S_i > M:
            u_hat = i
            M = sum2_S_i
    return u_hat, M


def block_coordinate_descent(Y_bar, lambda_, X_bar, gamma, max_iter=1000, eps=1e-4, verbose=0):
    """
    Implements Algorithm 1 of [1]_.

    Parameters
    ----------
    Y_bar : numpy.array of shape (n=n_samples, p=n_features)
        The centered signal.
    lambda_ : non-negative float
        The Lasso penalty coefficient.
    X_bar : numpy.array of shape (n, n-1)
        See referenced algorithm.
    gamma : numpy.array of shape (n-1,)
        See referenced algorithm.
    max_iter : positive int
        The maximum number of iterations.
    eps : non-negative number
        The threshold at which a float is considered non-null.
    verbose : int
        The verbosity level.


    Returns
    -------
    beta : numpy.array of shape (n-1, p)
        The solution of the group fused Lasso.
    KKT : bool
        `True` if the KKT conditions are verified, `False` else.
    niter : int
        The number of performed iterations by the algorithm.

    .. [1] Jean-Philippe Vert, Kevin Bleakley: The group fused Lasso for multiple change-point detection. _CoRR abs/1106.4199_ (2011)
    """
    tic = dt.now()

    # Check parameters
    try:
        n, p = Y_bar.shape
    except ValueError:
        raise ValueError('Y_bar must have 2 dimensions but has {}'.format(Y.ndim))
    if not isinstance(lambda_, Number) or lambda_ < 0:
        raise ValueError('lambda_ must be a non-negative number')
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError('max_iter must be a positive integer')
    if not isinstance(eps, Number) or eps < 0:
        raise ValueError('eps must be a positive number')
    if not isinstance(verbose, int) and verbose < 0:
        raise ValueError('verbose must be a positive int')

    # Init
    A = []
    beta = np.zeros((n-1, p))
    C = X_bar.T.dot(Y_bar)
    S = np.empty_like(C)
    lambda_square = lambda_ ** 2
    KKT = False

    # Loop
    for niter in range(1, max_iter + 1):
        # Block coordinate descent
        convergence = False
        A_shuffled = rdm.permutation(A).tolist()
        if verbose >= 1:
            print(80 * "-")
        while not convergence and A_shuffled:
            i = A_shuffled.pop()
            if verbose >= 1:
                print('time={}\tniter={}\ti={}\tbeta_i={}'.format(dt.now() - tic, niter, i, beta[i, :].round(3)))
            not_i = [j for j in range(beta.shape[0]) if j != i]
            S[i, :] = C[i, :] - X_bar[:, i].T.dot(X_bar[:, not_i].dot(beta[not_i, :]))
            beta[i, :] = update_beta(S[i, :], lambda_, gamma[i])
            convergence = check_kkt(S[A, :], beta[A, :], lambda_, eps)
        A = [i for i in A if npl.norm(beta[i, :]) >= eps]  # Remove inactive groups
        # Check global KKT
        S = C - X_bar.T.dot(X_bar.dot(beta))
        u_hat, M = compute_u_hat_and_M(S, A)
        if M > lambda_square:
            A.append(u_hat)
        else:
            KKT = True
        if KKT:
            break

    if verbose >= 1:
        print('KKT:', KKT)
    return beta, KKT, niter


def gflasso(Y, lambda_, max_iter=1000, eps=1e-4, center_Y=True, verbose=0):
    """
    Solves the group fused Lasso via a block coordinate descent algorithm [1]_.

    Parameters
    ----------
    Y : numpy.array of shape (n, p)
        The signal.
    lambda_ : non-negative float
        The Lasso penalty coefficient.
    max_iter : positive int
        The maximum number of iterations.
    eps : non-negative number
        The threshold at which a float is considered non-null.
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
    block_coordinate_descent()

    .. [1] Jean-Philippe Vert, Kevin Bleakley: The group fused Lasso for multiple change-point detection. _CoRR abs/1106.4199_ (2011)
    """
    # Check parameters
    if Y.ndim == 1:
        Y_bar = Y.reshape(-1, 1)
    elif Y.ndim == 2:
        Y_bar = Y
    else:
        raise ValueError('Y must have 1 or 2 dimensions but has {}'.format(Y.ndim))
    if not isinstance(lambda_, Number) or lambda_ < 0:
        raise ValueError('lambda_ must be a non-negative number')
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError('max_iter must be a positive integer')
    if not isinstance(eps, Number) or eps < 0:
        raise ValueError('eps must be a positive number')

    # Perform block coordinate descent
    n, p = Y_bar.shape
    if center_Y:
        Y_bar = scale(Y_bar, with_std=False)
    X = np.zeros((n, n-1))
    for i in range(n):
        n_, i_ = n+1, i+1
        X[i, :i] = np.sqrt(n_ / (i_ * (n_ - i_)))
    X_bar = scale(X, with_std=False)
    gamma = lambda x : x.T.dot(x)
    gamma = np.apply_along_axis(gamma, 0, X_bar)
    beta, KKT, niter = block_coordinate_descent(Y_bar, lambda_, X_bar, gamma, max_iter, eps, verbose)
    gamma = np.ones((1, n)).dot(Y - X.dot(beta)) / n  # Be careful, it is not the same gamma as in Algorithm 1, see Eq. (7) instead
    U = np.ones((n, 1)).dot(gamma) + X.dot(beta)

    return beta, KKT, niter, U


def breakpoints(beta, n=-1, eps=1e-4):
    """
    Post-processes the `beta` of the group fused Lasso [1]_ to get the breakpoints of a signal.
    The breakpoints are given by sorting the norms of the rows of `beta`.

    Parameters
    ----------
    beta : numpy.array of shape (n, p)
        The `beta` of the group fused Lasso.
    n : int
        The maximum number of breakpoints to retrieve. If negative, return all.
    eps : non-negative number
        The threshold at which a float is considered non-null.

    Returns
    -------
    list of int
        The breakpoints indexes.

    .. [1] Jean-Philippe Vert, Kevin Bleakley: The group fused Lasso for multiple change-point detection. _CoRR abs/1106.4199_ (2011)
    """
    bpts = []
    beta_norm = np.apply_along_axis(npl.norm, 1, beta)
    if n < 0:
        n = beta.shape[0]
    bpts = np.argsort(beta_norm)[-1:-(n+1):-1]
    bpts = [i for i in bpts if beta_norm[i] >= eps]
    return bpts


def plot_gflasso(Y, beta, bpts_pred=None, bpts_true=None, U=None):
    """
    Plots the results of the group fused Lasso with `beta` and the breakpoints.

    Parameters
    ----------
    Y : numpy.array of shape (n, p)
        The multidimensional signal.
    beta : numpy.array of shape (n-1, p)
        The group fused Lasso coefficients.
    nbpts : int
        The number of breakpoints to plot. If negative, plots them all.
    bpts_true : list of list of int
        The true breakpoints.
    U : numpy.array of shape (n, p)
        The multidimensional reconstructed signal.

    Returns
    -------
    fig, axs : matplotlib.figure.Figure, numpy.array of matplotlib.axes.Axes
        The matplotlib figure and axes objects given by `matplotlib.pyplot.subplots()` to plot results.
    """
    if Y.ndim == 1:
        Y_ = Y.reshape(-1, 1)
    elif Y.ndim == 2:
        Y_ = Y
    n, p = Y_.shape
    figsize = plt.rcParams['figure.figsize']
    figsize[0] *= p / 2
    fig, axs = plt.subplots(2, p, sharex=True, figsize=figsize)
    for j in range(p):
        ax = axs[0, j]
        j_ = j + 1
        ax.plot(range(n), Y_[:, j], '.', label='$Y_{:%d}$' % j_)
        if U is not None:
            ax.plot(range(n), U[:, j], '.', label='$U_{:%d}$' % j_)
            ax.set_ylabel('$Y_{:%d}, U_{:%d}$' % (j_, j_))
        else:
            ax.set_ylabel('$Y_{:%d}$' % j_)

        if bpts_true is not None:
            for i, b in enumerate(bpts_true):
                kwargs = {'color': plt.cm.tab10(2), 'ls': '--'}
                if i == 0:
                    ax.axvline(b, label='true bpts', **kwargs)
                else:
                    ax.axvline(b, **kwargs)
        if bpts_pred is not None:
            for i, b in enumerate(bpts_pred):
                kwargs = {'color': 'k', 'ls': '--'}
                if i == 0:
                    ax.axvline(b, label='pred bpts', **kwargs)
                else:
                    ax.axvline(b, **kwargs)
                axs[1, j].axhline(beta[b, j], color='k', ls='--')

        ax = axs[1, j]
        ax.stem(range(n), [np.nan] + beta[:, j].tolist(), markerfmt='.')
        ax.set_xlabel('$t$')
        ax.set_ylabel(r'$\beta_{:%d}$' % j_)

        for ax in axs.ravel():
            ax.set_xlim([0, n-1])

    fig.tight_layout()

    return fig, axs


if __name__ == '__main__':
    n, p = 400, 3
    Y = np.empty((n, p))
    nbpts = 3
    bpts_true = []
    mu, sigma = 5 * rdm.randn(), rdm.randn()
    Y = mu + sigma * rdm.randn(n, p)
    bpts_true = sorted(rdm.permutation(n)[:3])
    for j in range(p):
        for i in range(2):  # sorted(rdm.permutation(n)[:nbpts-1]):
            mu, sigma = 5 * rdm.randn(), rdm.randn()
            Y[bpts_true[i]:bpts_true[i+1], j] += mu + sigma * rdm.randn(bpts_true[i+1] - bpts_true[i])

    eps = 1e-8
    beta, KKT, niter, U = gflasso(Y=Y, lambda_=20, max_iter=1000, eps=eps, verbose=1)
    bpts_pred = breakpoints(beta, nbpts)
    plot_gflasso(Y, beta, bpts_pred, bpts_true, U)
    plt.show()
