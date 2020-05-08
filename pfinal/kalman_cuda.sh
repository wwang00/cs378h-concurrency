#!/bin/bash

DATA=EWA-EWC

IFILE=$PWD/input/$DATA-input.txt
OFILE=$PWD/output/$DATA-output.txt
rm -f $OFILE

cd workspace
make kalman_cuda

./kalman_cuda -i $IFILE -o $OFILE

cd ..
