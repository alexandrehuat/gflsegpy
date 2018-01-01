# -*- coding: utf-8 -*-

"""
:Author: Alexandre Huat <alexandre.huat@gmail.com>

This module is a demonstration on how to use the pygfl package.

In a Python 3 environment, use `python -m gfl.demo` to run it.
It will plot the results of the GFL block coordinate descent and the GFL LARS
on a simulated signal with gaussian noise.
You can monitor the hyperparameters of each parameter via the script arguments; run `python -m gfl.demo -h` to see all options.
"""

import argparse
import numpy.random as rdm
import matplotlib.pyplot as plt
plt.ion()
from gfl import gfl_coord, find_breakpoints, plot_breakpoints, gfl_lars

def _signal(shape=(500, 3), nbpts=4):
    n, p = shape
    musigma = lambda : (5 * rdm.randn(), rdm.randn())
    mu, sigma = musigma()
    Y = mu + sigma * rdm.randn(n, p)
    bpts = sorted(rdm.permutation(n)[:nbpts])
    for j in range(p):
        for i in range(nbpts-1):
            mu, sigma = musigma()
            Y[bpts[i]:bpts[i+1], j] += mu + sigma * rdm.randn(bpts[i+1] - bpts[i])
    return Y, bpts

    # step = 300
    # offset = 30000
    # Y = wav.read("data/signal_audio_1.wav")[1][offset::step, :]
    # bpts = (np.array([220500, 441000]) / step + offset).round().astype(int)
    # nbpts = len(bpts)
    # n, p = Y.shape


def _plot(title, Y, bpts_pred, bpts_true, beta=None, U=None):
    figs = plot_breakpoints(Y, bpts_pred, bpts_true, beta, U)
    for f in figs:
        ax = f.get_axes()[0]
        if ax.get_title():
            ax.set_title(title + "\n" + ax.get_title())
        else:
            ax.set_title(title)
        f.tight_layout()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--bpts_true", type=int, default=4,
                        help="the number of true breakpoints (default: 4)")
    parser.add_argument("-b", "--bpts_pred", type=int, default=4,
                        help="the number of breakpoints to find (default: 4)")
    parser.add_argument("-C", "--coord", action="store_true", default=False,
                        help="do not run the block coordinate descent (default: False)")
    parser.add_argument("-l", "--lam", type=float, default=10,
                        help="the lambda for GFL block coordinate descent (default: 10)")
    parser.add_argument("-L", "--lars", action="store_true", default=False,
                        help="do not run the LARS (default: False)")
    parser.add_argument("-s", "--shape", nargs=2, type=int, default=[500, 3],
                        help="the shape of the signal (default: (500, 3))")
    parser.add_argument("-I", "--max_iter", type=int, default=100,
                        help="the maximum iterations performed by each algorithm (default: 100)")
    parser.add_argument("-e", "--eps", type=float, default=1e-6,
                        help="the threshold at which a float is considered (default: 1e-6)")
    parser.add_argument("-v", "--verbose", action="store_true", default=True,
                        help="enable verbosity")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print("Demo params:", str(args)[:-1].replace("Namespace(", ""), end=2*"\n")

    Y, bpts_true = _signal(args.shape, args.bpts_true)

    # Apply GFL block coordinate descent
    if not args.coord:
        beta, KKT, niter, U = gfl_coord(Y=Y, lambda_=args.lam, max_iter=args.max_iter, eps=args.eps, verbose=int(args.verbose))
        bpts_pred = find_breakpoints(beta, args.bpts_pred)
        _plot("GFL block coordinate descent demo", Y, bpts_pred, bpts_true, beta, U)

    print()

    # Apply GFL LARS
    if not args.lars:
        bpts_pred = gfl_lars(Y, args.bpts_pred, verbose=int(args.verbose))
        _plot("GFL LARS demo", Y, bpts_pred, bpts_true)

    # plt.show()
    print()
    print("Wait for plots...")
    print("Press Enter to quit.")
    input()
    plt.close("all")
