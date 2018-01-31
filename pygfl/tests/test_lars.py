import pytest
import numpy as np
import numpy.random as rdm
from ..lars import gfl_lars

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


@pytest.mark.skip(reason="not ready")
def test_gfl_lars():
    """
    Test the shape of the breakpoints returned by `gfl_lars()`.
    """
    bpts = gfl_lars(Y(), 3, center_Y=True)
    assert len(bpts) == 3
    assert len(set(bpts)) == 3, "Some breakpoints are equals: {}".format(bpts)
