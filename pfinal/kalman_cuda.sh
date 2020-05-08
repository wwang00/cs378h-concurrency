#!/bin/bash

DATA=random-100-1000

IFILE=$PWD/input/$DATA-input.txt
OFILE=$PWD/output/$DATA-output.txt
rm -f $OFILE

cd workspace
make kalman_cuda

./kalman_cuda -i $IFILE -o $OFILE -g 16 -b 128

cd ..
