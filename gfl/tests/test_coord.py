import pytest
import numpy as np
import numpy.random as rdm
from ..coord import _block_coordinate_descent, gfl_coord, find_breakpoints

N, P = 60, 3

@pytest.fixture
def Y():
    """
    Returns a centered simulated signal of shape (60, 3) with 3 breakpoints.
    """
    Y = 1 * rdm.randn(N // 3, P) + 7
    Y = np.concatenate([Y, 2 * rdm.randn(N // 3, P) + 0])
    Y = np.concatenate([Y, 0.5 * rdm.randn(N // 3, P) + 6])
    Y -= np.outer(np.ones(N), Y.mean(axis=0))
    return Y


def test_block_coordinate_descent():
    """
    Test the shape of the `beta` returned by `block_coordinate_descent()`.
    """
    beta, KKT, niter = _block_coordinate_descent(Y(), 0.1, max_iter=100, eps=1e-4)
    assert beta.shape[0] == N - 1
    assert beta.shape[1] == P
    assert KKT or niter > 1


def test_gfl_coord():
    """
    Test the outputs formats of `gflasso()`.
    """
    beta, KKT, niter, U = gfl_coord(Y(), 0.1, max_iter=100, eps=1e-4, center_Y=True)

    assert beta.shape[0] == N - 1
    assert beta.shape[1] == P
    assert KKT or niter > 1
    assert U.shape[0] == N
    assert U.shape[1] == P


def test_find_breakpoints():
    beta = rdm.rand(N-1, P)

    n = 4
    bpts = find_breakpoints(beta, n, min_step=0, eps=0)
    assert len(bpts) == n

    n = beta.shape[0]
    bpts = find_breakpoints(beta, -1, min_step=0, eps=0)
    assert len(bpts) == n

    bpts = find_breakpoints(beta, -1, min_step=2, eps=0)
    assert len(bpts) < n
