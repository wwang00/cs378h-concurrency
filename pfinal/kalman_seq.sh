#!/bin/bash

DATA=random-150-120

IFILE=$PWD/input/$DATA-input.txt
OFILE=$PWD/output/$DATA-output.txt
rm -f $OFILE

cd workspace
make -s kalman_seq

./kalman_seq -i $IFILE -o $OFILE

cd ..
