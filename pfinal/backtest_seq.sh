#!/bin/bash

DATA=EWA-EWC

IFILE=$PWD/input/$DATA-input.txt
OFILE=$PWD/output/$DATA-output.txt
TESTS=5
rm -f $OFILE

cd workspace
make backtest_seq

./backtest_seq -i $IFILE -t $TESTS

cd ..
