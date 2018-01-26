# -*- coding: utf-8 -*-

from datetime import datetime as dt
import numpy as np


def col_sumsq(X):
    return (X ** 2).sum(axis=1)


def hstack(x, p):
    """
    Horizontally stacks `p` times the column vector `x`.

    Returns
    -------
    x_matrix : numpy.array of shape (x.size, p)
    """
    return np.outer(x, np.ones(p, dtype=bool))


def vstack(x, n):
    """
    Vertically stacks `n` times the row vector `x`.

    Returns
    -------
    x_matrix : numpy.array of shape (n, x.size)
    """
    return np.outer(np.ones(n, dtype=bool), x)


def center_matrix(Y):
    return Y - vstack(Y.mean(axis=0), Y.shape[0])