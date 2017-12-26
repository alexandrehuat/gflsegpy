import numpy as np

def center_matrix(Y):
    return Y - np.outer(np.ones(Y.shape[0]), Y.mean(axis=0))

def row_sumsq(X):
    return (X ** 2).sum(axis=1)
