"""
Author: Alexandre Huat
"""
from .base import GroupFusedLasso
import numpy as np


class GFLCW(GroupFusedLasso):
    def __init__(self):
        raise NotImplementedError

    def detect(self, Y, **kwargs):
        """
        Detect the breakpoints in Y.
        """
        n, p = Y.shape
        active = []
        beta = np.zeros(n, p)
