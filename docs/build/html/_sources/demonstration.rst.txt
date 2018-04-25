Demonstration
=============

Here is a demonstration on a 10-dimensionnal signal of length 500. You can run the module `gflsegpy.demo` to
reproduce a similar demonstration.

The segmented signal is a random gaussian signal on which has been applied two random gaussian noises at position 290
and 391 (the signal starts at position 0). The noises are independant from one dimension to another.

The figures below show the results of the gflsegpy algorithms.
To ensure readability, only the first three dimensions of the signal are plotted.

Algorithm 1: Block coordinate descent
-------------------------------------

This is the demonstration of the group fused Lasso block coordinate descent.
All three figures below show:
   * on top: the original signal in blue, the reconstructed signal in orange, the true breakpoints in magenta and
     the detected breakpoints in dashed black;
   * at the bottom: :math:`\beta` the matrix of the Lasso coefficients (the larger :math:`\beta_{i,j}`, the more likely
     :math:`i` is a change-point in the :math:`j`:sup:`th` dimension).
The true breakpoints are sorted in increasing order whereas the predicted breakpoints are sorted in order of importance according to the algorithm.

.. _c1:
.. figure:: demo_gfl_coord_1.png
   :scale: 70
   :align: left

   The first dimension of the signal

.. _c2:
.. figure:: demo_gfl_coord_2.png
   :scale: 70
   :align: center

   The second dimension of the signal

.. _c3:
.. figure:: demo_gfl_coord_3.png
   :scale: 70
   :align: center

   The third dimension of the signal

First of all, since the predicted breakpoints are the true ones, the optimum has been reached by this algorithm.
The computation time was 2.34 seconds. The block coordinate descent is accurate, but slow.

Secondly, for all position :math:`i` and :math:`j`:sup:`th` dimension, we can see that :math:`\beta_{i,j}` correlates with
the sign and the magnitude of the corresponding jump.

Eventually, remember that, in this setting, what really accounts for a change-point at :math:`i` is
:math:`\Vert\beta_{i,\bullet}\Vert`. This enables to capture the multidimensionality of the signal.
:numref:`c1` illustrates it well as :math:`\beta_{290,1}` is really small, but the change-point is finally detected via
a big :math:`\Vert\beta_{290,\bullet}\Vert`.


Algorithm 2: LARS
-------------------------------------

This is the demonstration of the group fused LARS on the same signal.

**TODO**