#!/bin/bash

DATA=EWA-EWC
TESTS=10

IFILE=$PWD/input/$DATA-input.txt

cd workspace
make backtest_cuda

./backtest_cuda -i $IFILE -t $TESTS

cd ..
