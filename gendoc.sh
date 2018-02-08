#!/bin/sh
cd docs
sphinx-apidoc -f -o source ../gflsegpy
make html
cd ..
open docs/build/html/index.html