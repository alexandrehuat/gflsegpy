"""
:Author: Alexandre Huat <alexandre.huat@gmail.com>

This module contains functions to plot the results of the group fused Lasso:
signal and breakpoints visualization
"""

from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl

MAX_PLOT = 3


def _bpts_title(bpts_pred=None, bpts_true=None):
    printbp = lambda bpts : sorted(bpts)  # ", ".join(str(b) for b in sorted(bpts))
    title = ""
    if bpts_pred is not None:
        title += "bpts_pred={}".format(printbp(bpts_pred))
    if bpts_true is not None:
        if title:
            title += "\nbpts_true={}".format(printbp(bpts_true))
        else:
            title = "bpts_true={}".format(printbp(bpts_true))
    return title


def plot_breakpoints(Y, bpts_pred=None, bpts_true=None, beta=None, U=None):
    """
    Plots the results of the group fused Lasso with the breakpoints and (optionally) `beta`.

    Parameters
    ----------
    Y : numpy.array of shape (n, p)
        The signal.
    nbpts : int
        The number of breakpoints to plot. If negative, plots them all.
    bpts_true : list of list of int
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
    if Y.ndim == 1:
        Y_ = Y.reshape(-1, 1)
    elif Y.ndim == 2:
        Y_ = Y
    n, p = Y_.shape
    figsize = np.array(plt.rcParams["figure.figsize"])
    nrows = 1
    if beta is not None:
        figsize[1] *= 1.5
        nrows = 2
        beta_norm = np.apply_along_axis(npl.norm, 1, beta)
    if p > MAX_PLOT:
        warn("Signal Y has more than {0} dimensions; only the {0} first will be plotted.".format(MAX_PLOT), UserWarning)
    figs = []
    for j in range(min([p, MAX_PLOT])):
        fig = plt.figure(figsize=figsize)
        xx = range(1, n+1)

        # Plot Y
        ax = fig.add_subplot(nrows, 1, 1)
        j_ = j + 1
        ax.plot(xx, Y_[:, j], ".", label=r"$Y_{\bullet,%d}$" % j_)
        if U is not None:
            ax.plot(xx, U[:, j], ".", label=r"$U_{\bullet,%d}$" % j_, alpha=0.5)
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
                kwargs = {"color": "k" if bpts_true else "k", "ls": "--"}
                if i == 0:
                    ax.axvline(b, label="bpts_pred", **kwargs)
                else:
                    ax.axvline(b, **kwargs)
        ax.legend(ncol=2)

        # Plot beta
        if beta is not None:
            ax = fig.add_subplot(nrows, 1, 2, sharex=ax)
            for _ in range(min([p, MAX_PLOT])):
                ax.bar(xx, [np.nan] + beta[:, _].tolist(), alpha=0.5, label=r"$\beta_{\bullet,%d}$" % (_+1))
            ax.plot(xx, [np.nan] + beta_norm.tolist(), label=r"$(||\beta_{i,\bullet}||)_{i=1,…,n-1}$")
            ax.grid(axis="y")
            ax.legend(ncol=2)
            ax.set_title(_bpts_title(bpts_true, bpts_pred))

        ax.set_title(_bpts_title(bpts_pred, bpts_true))
        ax.set_xlabel("$i$")
        ax.set_xlim([1, n])
        fig.tight_layout()
        figs.append(fig)
    return figs
