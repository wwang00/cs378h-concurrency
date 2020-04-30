#!/bin/bash

N_PTS=1000

IFILE=$PWD/input/input-$N_PTS.txt
OFILE=$PWD/output/output-$N_PTS.txt
rm -f $OFILE

# python3 rng.py $N_PTS $IFILE

cd workspace
make all

./kalman_seq -i $IFILE -o $OFILE -p $N_PTS

cd ..
