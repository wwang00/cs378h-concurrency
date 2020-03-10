#!/bin/bash

REPS=10
PROGS="cuda"
CFGS="32 128 512 2048"
N=65536
D=32
K=16
M=150
T="1e-5"
S=8675309

cd workspace
make clean
for prog in $PROGS; do
    make $prog
done
cd ..

rm output/*.data

for prog in $PROGS; do
    for blks in 1; do
	for tpb in 16384 32768; do
	    for (( i = 0; i < REPS; i++ )); do
		id="n${N}-d${D}"
		ifile="../input/${id}.txt"
		ofile="../output/${id}-b${blks}-t${tpb}-${prog}.data"
		echo $id - $blks - $tpb - $prog - $i
		cd workspace
		./$prog -k $K -d $D -i $ifile -m $M -t $T -s $S -blks $blks -tpb $tpb >> $ofile
		cd ..
	    done
	done
    done
done
