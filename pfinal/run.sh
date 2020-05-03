#!/bin/bash

N_PTS=1250

S1=EWA
S2=EWC

IFILE1=$PWD/input/$S1-input.txt
IFILE2=$PWD/input/$S2-input.txt
OFILE=$PWD/output/$S1-$S2-output.txt
rm -f $OFILE

cd workspace
make all

./kalman_seq -i1 $IFILE1 -i2 $IFILE2 -o $OFILE -p $N_PTS

cd ..
