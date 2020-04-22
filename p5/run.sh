#!/bin/bash

FILE=nb-10
IFILE=$PWD/input/$FILE.txt
OFILE=$PWD/output/$FILE.txt

cd workspace
make all

mpiexec -n 1 ./nbody -i $IFILE -o $OFILE -s 1 -t 0.86010 -d 0.005

cd ..
