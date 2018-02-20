.. gflsegpy documentation master file, created by
   sphinx-quickstart on Thu Feb  8 20:26:44 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to gflsegpy's documentation!
====================================

gflsegpy is a full Python 3 implementation of the group fused Lasso (GFL) for multiple change-points detection [1]_.
It is largely based upon `NumPy <http://www.numpy.org>`_,
while the original MATLAB implementation of the authors is available at http://cbio.ensmp.fr/GFLseg.

    "**Abstract**—*We present the group fused Lasso for detection of multiple change-points shared by a set of
    co-occurring one-dimensional signals. Change-points are detected by approximating the original signals with a
    constraint on the multidimensional total variation, leading to piecewise-constant approximations. Fast algorithms
    are proposed to solve the resulting optimization problems, either exactly or approximately. Conditions are given
    for consistency of both algorithms as the number of signals increases, and empirical evidence is provided to
    support the results on simulated and array comparative genomic hybridization data.*"

Below lays the contents of the documentation.

Please, if you use the gflsegpy package, cite it:

* Plain text:

    A. Huat, *gflsegpy: A Python 3 implementation of the group fused Lasso for multiple change-point detection*,
    `https://github.com/alexandrehuat/gflsegpy`, GitHub repository, Feb. 2018.

* BibTex::

    @misc{gflsegpy,
        author = {Huat, Alexandre},
        title = {{gflsegpy}: A {Python} 3 implementation of the group fused
        {Lasso} for multiple change-point detection},
        howpublished = {https://github.com/alexandrehuat/gflsegpy},
        type = {GitHub repository},
        month = {2},
        year = {2018},
    }

References
""""""""""
.. [1] K. Bleakley and J.-P. Vert, “The group fused Lasso for multiple change-point detection”,
       *ArxXiv e-prints*, Jun. 2011. arXiv: `1106.4199 [q-bio.QM]
       <https://arxiv.org/abs/1106.4199>`_.

Contents
========

.. toctree::
   :maxdepth: 3

   versions
   requirements
   demonstration
   usage
   bugs

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
