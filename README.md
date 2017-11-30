# The group fused Lasso for multiple change-point detection
## A Python 3 implementation

__Alexandre Huat, Joseph Lefevre__ (INSA Rouen Normandie)

----

This repository is a Python 3 implementation of the group fused Lasso for multiple change-point detection, according to the paper of [Bleakley and Vert, 2011][1].

> _Abstract_—We present the group fused Lasso for detection of multiple change-points shared by a set of co-occurring one-dimensional signals. Change-points are detected by approximating the original signals with a constraint on the multidimensional total variation, leading to piecewise-constant approximations. Fast algorithms are proposed to solve the resulting optimization problems, either exactly or approximately. Conditions are given for consistency of both algorithms as the number of signals increases, and empirical evidence is provided to support the results on simulated and array comparative genomic hybridization data.

The MATLAB implementation of the authors is available at http://cbio.ensmp.fr/GFLseg.

[1]: https://arxiv.org/abs/1106.4199
