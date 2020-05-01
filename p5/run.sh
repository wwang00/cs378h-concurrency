#!/bin/bash

FBASE=$PWD
FILE=nb-10
IFILE=$FBASE/input/$FILE.txt
OFILE=$FBASE/output/$FILE.out

rm -f $OFILE

cd workspace
make all

# mpiexec -n $1 ./nbody -i $IFILE -o $OFILE -s 1000 -t 0.35 -d 0.005

for n in 100 200 300 400 500; do
    FILE=nb-$n
    IFILE=$FBASE/input/$FILE.txt
    DFILE=$FBASE/data/$FILE-p.dat
    touch $DFILE
    rm -f $DFILE
    touch $DFILE
    for i in {1..10}; do
        echo $n $i
        mpiexec -n 4 ./nbody -i $IFILE -o /dev/null -s 10000 -t 0.35 -d 0.005 >> $DFILE
    done
done

cd ..
