# -*- coding: utf-8 -*-
"""
Usage: `python3 -m GFLsegpy.demo [-h] [-B BPTS_TRUE] [-b BPTS_PRED] [-s N P] [-L] [-C] [-l LAMBDA] [-m MIN_STEP] [-I MAX_ITER] [-e EPS] [-v]`

This module is a demonstration on gflsegpy on a gaussian random signal.

Optional arguments:
    -h, --help
        show this help message and exit
    -B BPTS_TRUE, --bpts_true BPTS_TRUE
        the number of true breakpoints (default: 2)
    -b BPTS_PRED, --bpts_pred BPTS_PRED
        the number of breakpoints to find (default: 2)
    -s N_SPACE_P, --shape N_SPACE_P
        the shape of the signal (default: (500, 3))
    -L, --lars
        run the LARS
    -C, --coorde
        run the block coordinate descent
    -l LAMBDA, --lam LAMBDA
        the :math:`\lambda` of the GFL block coordinate descent (default: 10)
    -m MIN_STEP, --min_step MIN_STEP
        the minimal step between two predicted breakpoints
    -I MAX_ITER, --max_iter MAX_ITER
        the maximum iterations performed by each algorithm (default: 100)
    -e EPS, --eps EPS
        the threshold at which a float is considered non-null (default: 1e-6)
    -v, --verbose
        the verbosity level (the more the number of `v`, the greater the verbosity)
"""

import argparse
import numpy as np
import numpy.random as rdm
import matplotlib.pyplot as plt
plt.ion()
from gflsegpy.coord import _gfl_coord, _find_breakpoints
from gflsegpy import plot_breakpoints, gfl_lars


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


def _parse_args():
    parser = argparse.ArgumentParser(prog="gflsegpy.demo",
            description="This module is a demonstration on a gaussian random signal.")
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
                        help="the lambda of the GFL block coordinate descent (default: 10)")
    parser.add_argument("-m", "--min_step", type=int, default=4,
                        help="the minimal step between two predicted breakpoints")
    parser.add_argument("-I", "--max_iter", type=int, default=100,
                        help="the maximum iterations performed by each algorithm (default: 100)")
    parser.add_argument("-e", "--eps", type=float, default=1e-6,
                        help="the threshold at which a float is considered non-null (default: 1e-6)")
    parser.add_argument("-v", "--verbose", action="count", default=1,
                        help="the verbosity level (the more the number of 'v', the greater)")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()

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
        bpts_pred = gfl_lars(Y, args.bpts_pred, verbose=args.verbose)
        _plot("GFL LARS", Y, bpts_pred, bpts_true)
        print()

    if args.coord:  # or args.lars:
        # plt.show()
        print("Press Enter to close all and quit.")
        input()
        plt.close("all")
