#!/bin/bash

FILE=nb-100
IFILE=$PWD/input/$FILE.txt
OFILE=$PWD/output/$FILE.out
DFILE_BASE=$PWD/data/

rm -f $OFILE
#rm -f $DFILE_BASE/*

cd workspace
make all

for n in {9..16}; do
    DFILE=$DFILE_BASE/$FILE-$n.dat
    touch $DFILE
    for i in {1..10}; do
	echo $n $i
	mpiexec -n $n ./nbody -i $IFILE -o $OFILE -s 10000 -t 0.35 -d 0.005 >> $DFILE
    done
done

cd ..
