# -*- coding: utf-8 -*-
"""
The documentation of all functions appears below (public and private),
but gflsegpy is very simple to use.
By importing :py:mod:`gflsegpy`, you will be able to call three top-level functions on which you
will have to focus as a user:
    * :py:func:`gfl_coord` from :py:mod:`gflsegpy.coord`;
    * :py:func:`gfl_lars` from :py:mod:`gflsegpy.lars`;
    * :py:func:`plot_breakpoints` from :py:mod:`gflsegpy.plot`.

Example
-------
>>> import gflsegpy as seg  # It won't work if you do 'from gflsegpy import *'
>>> import matplotlib.pyplot as plt
>>>
>>> ...  # Processing the signal to segmentate
>>> bpts = seg.gfl_coord(...)  # Perform a block coordinate descent on the signal
>>> ...
>>> bpts = seg.gfl_lars(...)  # Perform the GFL LARS on the signal
>>> ...
>>> seg.plot_breakpoints(...)  # Plot your results
>>> plt.show()  # Show the results
"""

from .coord import gfl_coord
from .lars import gfl_lars  # unstable
from .plot import plot_breakpoints

__all__ = ["coord", "lars", "plot"]