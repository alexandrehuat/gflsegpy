Introduction
============

gflsegpy is a full Python 3 implementation of the group fused Lasso (GFL) for multiple change-points detection,
as proposed by Bleakley and Vert [1]_.

    **Abstract**—We present the group fused Lasso for detection of multiple change-points shared by a set of
    co-occurring one-dimensional signals. Change-points are detected by approximating the original signals with a
    constraint on the multidimensional total variation, leading to piecewise-constant approximations. Fast algorithms
    are proposed to solve the resulting optimization problems, either exactly or approximately. Conditions are given
    for consistency of both algorithms as the number of signals increases, and empirical evidence is provided to
    support the results on simulated and array comparative genomic hybridization data.

While, gflsegpy is largely based upon `NumPy <http://www.numpy.org>`_,
the original MATLAB implementation of the authors is available at http://cbio.ensmp.fr/GFLseg.

Please, if you use my package, cite it:

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

Versions
--------

* v1.1.0 [dev]
   * GFL LARS (inexact but fast solution)
   * **TODO: Changeable weights**

* v1.0.0 (February DD, 2018) [stable]
   * GFL block coordinate descent (exact but slow solution)
   * Datavisualisation module
   * Demo script
   * Fixed weights scheme (see [1]_, Eq. (5))

Requirements
------------

It is highly recommended to run the package in a Python 3 virtual environment.
In such environment, you can install all requirements with pip by running::

    pip install -r requirements.txt

Requirements are the following:

.. literalinclude:: ../../requirements.txt
   :linenos:
   :language: text

Bugs report
-----------

If you encounter a malfunction, please, open an issue `here <https://github.com/alexandrehuat/gflsegpy/issues>`_.


References
----------

.. [1] K. Bleakley and J.-P. Vert, “The group fused Lasso for multiple change-point detection”,
       *ArxXiv e-prints*, Jun. 2011. arXiv: :download:`1106.4199 [q-bio.QM]
       <../2011-The_group_fused_Lasso_for_multiple_change-point_detection.pdf>`.
