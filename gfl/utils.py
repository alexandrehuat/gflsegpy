import numpy as np

def center_matrix(Y):
    return Y - np.outer(np.ones(Y.shape[0]), Y.mean(axis=0))
