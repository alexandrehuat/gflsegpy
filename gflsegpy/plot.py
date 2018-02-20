# -*- coding: utf-8 -*-
"""
This module provides some function to plot the results of the GFL,
allowing signal and breakpoints visualization.
"""


from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

ALPHA_TRANS = 2/3


def _bpts_title(bpts_pred=None, bpts_true=None):
    title = ""
    if bpts_pred is not None:
        title += "bpts_pred={}".format(bpts_pred.tolist())
    if bpts_true is not None:
        title += int(bool(title)) * "\n" \
                 + "bpts_true={}".format(bpts_true.tolist())
    return title


def plot_breakpoints(Y, bpts_pred=None, bpts_true=None, beta=None, U=None, max_dim=3):
    """
    Plots the signal `Y` and the results of the GFL:
    the breakpoints, `beta` the Lasso coefficients and `U` the reconstructed signal.

    Parameters
    ----------
    Y : numpy.array of shape (n, p)
        The signal.
    bpts_pred : optionnal, 1D-numpy.array of int
        The predicted breakpoints.
    bpts_true : optionnal, 1D-numpy.array of int
        The true breakpoints.
    U : optionnal, numpy.array of shape (n, p)
        The multidimensional reconstructed signal.
    beta : optionnal, numpy.array of shape (n-1, p)
        The GFL coefficients.
    max_dim : positive int
        The maximum number of dimensions to plot.

    Returns
    -------
    list of matplotlib.figure.Figure
        The matplotlib figure objects used to plot results.
    """
    # Arguments checking
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
    if max_dim > 5:
        raise UserWarning("More than 5 matplotlib figures will be plotted (one per signal dimension), "
                          "this may seriously slow down the system.")

    # Preparing plots
    nrows = 1
    if beta is not None:
        nrows = 2
        beta_norm = np.apply_along_axis(norm, 1, beta)
    if p > max_dim:
        warn("Signal Y has more than {0} dimensions; only the {0} first will be plotted.".format(max_dim), UserWarning)
    figs = []
    title = _bpts_title(bpts_pred, bpts_true)
    for j in range(min((p, max_dim))):
        fig = plt.figure()
        xx = range(n)  # x axis

        # Plot Y
        ax = fig.add_subplot(nrows, 1, 1)
        j_ = j + 1
        ax.plot(xx, Y[:, j], ".", label=r"$Y_{\bullet,%d}$" % j_)
        if U is not None:
            ax.plot(xx, U[:, j], ".", label=r"$U_{\bullet,%d}$" % j_, alpha=ALPHA_TRANS)
        ax.grid(axis="y")
        ax.set_title(title)

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
            if p > 1:
                ax.plot(xx[1:], beta[:, j], label=r"$\beta_{\bullet,%d}$" % j_)
            ax.plot(xx[1:], beta_norm, ":k", alpha=ALPHA_TRANS, label=r"$(\Vert\beta_{i,\bullet}\Vert)_{i=1}^{n-1}$")
            ax.grid(axis="y")
            ax.legend(ncol=2)

        ax.set_xlabel("$i$")
        ax.set_xlim((0, n-1))
        fig.tight_layout()
        figs.append(fig)

    return figs
