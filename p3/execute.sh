#!/bin/bash

REPS=20
PROGS="seq thrust"
CFGS=3
N=(2048 16384 65536)
D=(16 24 32)
K=16
M=150
T=1e-5
S=8675309

cd workspace
make clean
for prog in $PROGS; do
    make $prog
done
cd ..

rm output/*.data

for (( cfg = 0; cfg < CFGS; cfg++ )); do
    for prog in $PROGS; do
	for (( i = 0; i < REPS; i++ )); do
	    n=${N[$cfg]}
	    d=${D[$cfg]}
	    id="n${n}-d${d}"
	    ifile="../input/${id}.txt"
	    ofile="../output/${id}-${prog}.data"
	    echo $id - $prog - $i
	    cd workspace
	    ./$prog -k $K -d $d -i $ifile -m $M -t $T -s $S >> $ofile
	    cd ..
	done
    done
done
