# -*- coding: utf-8 -*-

"""
:Author: Alexandre Huat <alexandre.huat@gmail.com>

This module is a demonstration on how to use the gflsegpy package.

In a Python 3 environment, use `python -m gflsegpy.demo` to run it.
It will plot the results of the GFL block coordinate descent and the GFL LARS
on a simulated signal with gaussian noise.
You can monitor the hyperparameters of each parameter via the script arguments;
run `python -m gflsegpy.demo -h` to see all options.
"""

import argparse
import numpy as np
import numpy.random as rdm
import matplotlib.pyplot as plt
plt.ion()
from gflsegpy.coord import _gfl_coord, _find_breakpoints
from gflsegpy import plot_breakpoints  # , gfl_lars

def _signal(shape=(500, 3), nbpts=4):
    n, p = shape
    musigma = lambda : (5 * rdm.randn(), rdm.randn())
    mu, sigma = musigma()
    Y = mu + sigma * rdm.randn(n, p)
    bpts = np.array(sorted(rdm.permutation(n)[:nbpts]) + [n])
    for j in range(p):
        for i in range(nbpts):
            mu, sigma = musigma()
            Y[bpts[i]:bpts[i+1], j] += mu + sigma * rdm.randn(bpts[i+1] - bpts[i])
    return Y, bpts[:-1]

    # A true signal: a mix of three audio tracks
    # step = 300
    # offset = 30000
    # Y = wav.read("data/mix_audio_3.wav")[1][offset::step, :]
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
    parser = argparse.ArgumentParser(
        prog="gflsegpy.demo",
        description="This program demonstrates the results of the gflsegpy package "
                    "on a gaussian random signal.",
    )
    parser.add_argument("-B", "--bpts_true", type=int, default=2,
                        help="the number of true breakpoints (default: 2)")
    parser.add_argument("-b", "--bpts_pred", type=int, default=2,
                        help="the number of breakpoints to find (default: 2)")
    parser.add_argument("-s", "--shape", nargs=2, type=int, default=(500, 10),
                        help="the shape of the signal (default: (500, 3))")
    parser.add_argument("-L", "--lars", action="store_true", default=True,
                        help="run the LARS")
    parser.add_argument("-C", "--coord", action="store_true", default=True,
                        help="run the block coordinate descent")
    parser.add_argument("-l", "--lam", metavar="LAMBDA", type=float, default=10,
                        help="the lambda for the GFL block coordinate descent (default: 10)")
    parser.add_argument("-m", "--min_step", type=int, default=4,
                        help="the minimal step between two predicted breakpoints")
    parser.add_argument("-I", "--max_iter", type=int, default=100,
                        help="the maximum iterations performed by algorithms (default: 100)")
    parser.add_argument("-e", "--eps", type=float, default=1e-6,
                        help="the threshold at which a float is considered non-null (default: 1e-6)")
    parser.add_argument("-v", "--verbose", action="count", default=1,
                        help="the verbosity level (the more the number of 'v', the greater)")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    print("Demo params:", str(args)[:-1].replace("Namespace(", ""))
    print("Pyplot interactive plotting is on. Graphs will be drawed progressively.")
    print()

    Y, bpts_true = _signal(args.shape, args.bpts_true)
    print("True breakpoints:", bpts_true.tolist(), end=2*"\n")

    # Apply the GFL block coordinate descent
    if args.coord:
        beta, KKT, niter, U = _gfl_coord(Y=Y, lambda_=args.lam, max_iter=args.max_iter, eps=args.eps, verbose=args.verbose)
        bpts_pred = _find_breakpoints(beta, args.bpts_pred, args.min_step, verbose=args.verbose)
        _plot("GFL block coordinate descent ($\lambda={}$)".format(args.lam),
              Y, bpts_pred, bpts_true, beta, U)

        # In one step, but prevents from plotting beta and U
        # bpts_pred = gfl_coord(Y=Y, lambda_=args.lam, nbpts=args.bpts_pred, min_step=args.min_step,
        #                  max_iter=args.max_iter, eps=args.eps, verbose=args.verbose)
        # _plot("GFL block coordinate descent", Y, bpts_pred, bpts_true)

        print()

    # Apply the GFL LARS
    if args.lars:
        print("Oups... The group fused LARS is not released yet."
              "\nTry the block coordinate descent instead.")
        # bpts_pred = gfl_lars(Y, args.bpts_pred, verbose=args.verbose)
        # _plot("GFL LARS", Y, bpts_pred, bpts_true)
        print()

    if args.coord:  # or args.lars:
        # plt.show()
        print("Press Enter to close all and quit.")
        input()
        plt.close("all")
