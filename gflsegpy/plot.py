# -*- coding: utf-8 -*-

"""
:Author: Alexandre Huat <alexandre.huat@gmail.com>

This module contains some functions to plot the results of the group fused Lasso,
allowing signal and breakpoints visualization.
"""


from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

MAX_PLOT = 3


def _bpts_title(bpts_pred=None, bpts_true=None):
    title = ""
    if bpts_pred is not None:
        title += "bpts_pred={}".format(bpts_pred.tolist())
    if bpts_true is not None:
        title += int(bool(title)) * "\n" \
                 + "bpts_true={}".format(bpts_true.tolist())
    return title


def plot_breakpoints(Y, bpts_pred=None, bpts_true=None, beta=None, U=None):
    """
    Plots the signal `Y` and the results of the group fused Lasso:
    the breakpoints and (optionally) the Lasso coefficients `beta`.

    Parameters
    ----------
    Y : numpy.array of shape (n, p)
        The signal.
    bpts_pred : 1D-numpy.array of int
        The predicted breakpoints.
    bpts_true : 1D-numpy.array of int
        The true breakpoints.
    U : numpy.array of shape (n, p)
        The multidimensional reconstructed signal.
    beta : numpy.array of shape (n-1, p)
        The group fused Lasso coefficients.

    Returns
    -------
    fig, axs : matplotlib.figure.Figure, numpy.array of matplotlib.axes.Axes
        The matplotlib figure and axes objects given by `matplotlib.pyplot.subplots()` to plot results.
    """
    # Arguments check
    if Y.ndim != 2:
        raise ValueError("Y must be a numpy.array of shape (n, p) but is {}.".format(Y.shape))
    n, p = Y.shape
    if U is not None and any(Y.shape[i] != U.shape[i] for i in range(2)):
        raise ValueError("U must be a numpy.array of shape (n={}, p={}) but is {}.".format(n, p, U.shape))
    if beta is not None and beta.shape[0] != n-1 and beta.shape[1] != p:
        raise ValueError("beta must be a numpy.array of shape (n-1={}, p={}) but is {}.".format(n, p, beta.shape))
    if bpts_pred is not None and not isinstance(bpts_pred, np.ndarray):
        raise ValueError("bpts_pred must be a numpy.array.")
    if bpts_true is not None and not isinstance(bpts_pred, np.ndarray):
        raise ValueError("bpts_true must be a numpy.array.")

    # Preparing plots
    figsize = np.array(plt.rcParams["figure.figsize"])
    nrows = 1
    if beta is not None:
        figsize[1] *= 1.5
        nrows = 2
        beta_norm = np.apply_along_axis(norm, 1, beta)
    if p > MAX_PLOT:
        warn("Signal Y has more than {0} dimensions; only the {0} first will be plotted.".format(MAX_PLOT), UserWarning)
    figs = []
    for j in range(min((p, MAX_PLOT))):
        fig = plt.figure(figsize=figsize)
        xx = range(n)  # x axis

        # Plot Y
        ax = fig.add_subplot(nrows, 1, 1)
        j_ = j + 1
        ax.plot(xx, Y[:, j], ".", label=r"$Y_{\bullet,%d}$" % j_)
        if U is not None:
            ax.plot(xx, U[:, j], ".", label=r"$U_{\bullet,%d}$" % j_, alpha=2/3)
        ax.grid(axis="y")

        # Plot breakpoints
        if bpts_true is not None:
            for i, b in enumerate(bpts_true):
                kwargs = {"color": "m", "ls": "-"}
                if i == 0:
                    ax.axvline(b, label="bpts_true", **kwargs)
                else:
                    ax.axvline(b, **kwargs)
        if bpts_pred is not None:
            for i, b in enumerate(bpts_pred):
                kwargs = {"color": "k" if bpts_true.size else "k", "ls": "--"}
                if i == 0:
                    ax.axvline(b, label="bpts_pred", **kwargs)
                else:
                    ax.axvline(b, **kwargs)
        ax.legend(ncol=2)

        # Plot beta
        if beta is not None:
            ax = fig.add_subplot(nrows, 1, 2, sharex=ax)
            xx = xx[1:]
            if p > 1:
                for _ in range(min((p, MAX_PLOT))):
                    ax.bar(xx, beta[:, _], alpha=0.5, label=r"$\beta_{\bullet,%d}$" % (_+1))
            ax.plot(xx, beta_norm, label=r"$(\Vert\beta_{i,\bullet}\Vert)_{i=1}^{n-1}$")
            ax.grid(axis="y")
            ax.legend(ncol=2)
            ax.set_title(_bpts_title(bpts_true, bpts_pred))

        ax.set_title(_bpts_title(bpts_pred, bpts_true))
        ax.set_xlabel("$i$")
        ax.set_xlim((0, n-1))
        fig.tight_layout()
        figs.append(fig)
    return figs
