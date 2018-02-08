# gflsegpy: A Python 3 implementation of the group fused Lasso for multiple change-point detection
__Alexandre Huat__ (INSA Rouen Normandie)

----

gflsegpy is a full Python 3 implementation of the group fused Lasso (GFL) for multiple change-points detection [1].

> _Abstract_—We present the group fused Lasso for detection of multiple change-points shared by a set of co-occurring one-dimensional signals. Change-points are detected by approximating the original signals with a constraint on the multidimensional total variation, leading to piecewise-constant approximations. Fast algorithms are proposed to solve the resulting optimization problems, either exactly or approximately. Conditions are given for consistency of both algorithms as the number of signals increases, and empirical evidence is provided to support the results on simulated and array comparative genomic hybridization data.

While, gflsegpy is largely based upon NumPy, the original MATLAB implementation of the authors is available at [http://cbio.ensmp.fr/GFLseg]().

Please, if you use my package, cite it:
* Plain text

A. Huat, _gflsegpy: A Python 3 implementation of the group fused Lasso for multiple change-point detection_, `https://github.com/alexandrehuat/gflsegpy`, GitHub repository, Feb. 2018.

* BibTex

```bib
@misc{gflsegpy,
	author = {Huat, Alexandre},
	title = {gflsegpy: A Python 3 implementation of the group fused Lasso for multiple change-point detection},
	howpublished = {https://github.com/alexandrehuat/gflsegpy},
	type = {GitHub repository},
	month = {2},
	year = {2018},
}
```

## Version control

###### v1.1 [dev]

* GFL LARS (inexact but fast solution)
* TODO: Changeable weights

###### v1.0 [stable] (TODO: date)
* GFL block coordinate descent (exact but slow solution)
* Datavisualisation module
* Demo script
* Fixed weights scheme (see [1], Eq. (5))

----

## Requirements

Please, refer to the [`requirements.txt`](requirements.txt) file.

In a virtual environment, you can install all requirements with pip by running:

```sh
pip install -r requirements.txt
```

## Usage

This section gives an overview of the gflsegpy package. Read the documentation for more details.

The documentation is built with Sphinx.
To do so, get in the `docs` directory, generate the source files from the code and run `make html`. TL;DR:

### Algorithm 1: Block coordinate descent

The first algorithm proposed by Bleakley and Vert returns ![beta*](https://latex.codecogs.com/gif.latex?%5Cbeta^*)
the optimal coefficients of the GFL found via a block coordinate descent.
This algorithm gives an exact solution but is very slow in comparison with the LARS algorithm.

* For a rapid use, call the function `gflsegpy.gfl_coord()` which will return the desired number of change-points.
* For an advanced use, firstly call the function `gflsegpy.coord._gfl_coord()` which will return ![beta*](https://latex.codecogs.com/gif.latex?%5Cbeta^*).
Then, process ![beta*](https://latex.codecogs.com/gif.latex?%5Cbeta^*) to extract the change-points.
_N.B. I defined the function `gflsegpy.coord._find_breakpoints()` to perform such post-processing.
As you guess, `gflsegpy.gfl_coord()` simply call both of the method mentionned above._

The key parameter of the block coordinate descent is ![lambda](https://latex.codecogs.com/gif.latex?%5Clambda) the regularization coefficient of the GFL.
Note that the smaller ![lambda](https://latex.codecogs.com/gif.latex?%5Clambda), the greater the number of searched change-points;
_i.e._ the longer it will take to the algorithm to converge. Consider this when optimizing ![lambda](https://latex.codecogs.com/gif.latex?%5Clambda).

Then, an appropriate optimisation strategy would test a sequence of ![lambda](https://latex.codecogs.com/gif.latex?%5Clambda)
in decreasing order and stop when the validation error becomes _unacceptable_ or doesn't drop.

As well as tuning ![lambda](https://latex.codecogs.com/gif.latex?%5Clambda), you may also want to set the optimal weights of each signal position in the model.
This function will be added in the next stable release. For now, only the _default_ weights can be used (see [1], Eq. (5)).
Be advised that if you want to detect a single change-point,
Bleakley and Vert proved that the default weights are the best (see [1], Theorem 3).
Else, I won't give any recommendation; please, refer to [1] or your own experience.

### Algorithm 2: The GFL LARS

__WARNING: The current implementation may be unstable.__

This second algorithm is very faster than the GFL block coordinate descent, however it is less accurate.

Simply call `gflsegpy.gfl_lars()` to use it. In a nutshell, its inputs are the signal and the number of breakpoints to detect. Then, it returns the estimated breakpoints.

### Data visualization

Use the function `gflsegpy.plot_breakpoints()` to visualize the salient results of the algorithms applied to your signal.

## Demo

Run the script `gflsegpy.demo` to reproduce the demonstration rendered below.

___TODO___

## Bugs report

___TODO___

## References

[1] [Kevin Bleakley, Jean-Philippe Vert: The group fused Lasso for multiple change-point detection. _CoRR abs/1106.4199_ (2011)](docs/2011-The_group_fused_Lasso_for_multiple_change-point_detection.pdf)
