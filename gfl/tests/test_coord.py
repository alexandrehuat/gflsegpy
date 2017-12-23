import pytest
import numpy as np
import numpy.random as rdm
from ..lasso import _block_coordinate_descent, gfl_coord, find_breakpoints

N, P = 60, 3

@pytest.fixture
def Y():
    """
    Returns a centered simulated signal of shape (60, 3).
    """
    Y = 1 * rdm.randn(N // 3, P) + 7
    Y = np.concatenate([Y, 2 * rdm.randn(N // 3, P) + 0])
    Y = np.concatenate([Y, 0.5 * rdm.randn(N // 3, P) + 6])
    Y -= np.outer(np.ones(N), Y.mean(axis=0))
    return Y


@pytest.fixture
def X_gamma():
    X = np.tril(np.ones((N, N-1)))
    X -= np.outer(np.ones(N), X.mean(axis=0))
    gamma = lambda x : x.T.dot(x)
    gamma = np.apply_along_axis(gamma, 0, X)
    return X, gamma


def test_block_coordinate_descent():
    """
    Test the shape of the `beta` returned by `block_coordinate_descent()`.
    """
    X, gamma = X_gamma()
    beta, KKT, niter = _block_coordinate_descent(Y(), 0.1, X, gamma, max_iter=100, eps=1e-4)
    assert beta.shape[0] == N - 1
    assert beta.shape[1] == P
    assert KKT or niter > 1


def test_gflasso():
    """
    Test the outputs formats of `gflasso()`.
    """
    beta, KKT, niter, U = gflasso(Y(), 0.1, max_iter=100, eps=1e-4, center_Y=True)

    assert beta.shape[0] == N - 1
    assert beta.shape[1] == P
    assert KKT or niter > 1
    assert U.shape[0] == N
    assert U.shape[1] == P


def test_breakpoints():
    beta = rdm.randn(N-1, P)

    n = 4
    bpts = breakpoints(beta, n)
    assert len(bpts) == n

    n = beta.shape[0]
    bpts = breakpoints(beta, -1)
    assert len(bpts) == n
