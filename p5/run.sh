#!/bin/bash

FILE=nb-10
IFILE=$PWD/input/$FILE.txt
OFILE=$PWD/output/$FILE.out

rm -f $OFILE
cd workspace
make all

mpiexec -n 2 ./nbody -i $IFILE -o $OFILE -s 1 -t 0.1 -d 0.005

cd ..
