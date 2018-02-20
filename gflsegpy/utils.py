# -*- coding: utf-8 -*-
"""
This module provides a few utilities to ease implementations.
"""


import numpy as np


def col_sumsq(X):
    return (X ** 2).sum(axis=1)


def hstack(x, p):
    return np.outer(x, np.ones(p, dtype=bool))


def vstack(x, n):
    return np.outer(np.ones(n, dtype=bool), x)


def center_matrix(Y):
    return Y - vstack(Y.mean(axis=0), Y.shape[0])