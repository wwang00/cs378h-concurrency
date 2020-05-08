#!/bin/bash

DATA=random-100-1000

IFILE=$PWD/input/$DATA-input.txt
OFILE=$PWD/output/$DATA-output.txt
rm -f $OFILE

cd workspace
make kalman_seq

./kalman_seq -i $IFILE -o $OFILE

cd ..
