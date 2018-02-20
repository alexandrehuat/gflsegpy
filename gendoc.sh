#!/bin/sh
if [[ $1 == "-a" ]]; then
    sphinx-apidoc -f -P -o docs/source gflsegpy gflsegpy/tests/*
fi
sphinx-build -b html docs/source docs/build/html
