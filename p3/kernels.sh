#!/bin/bash

REPS=10
PROGS="shmem"
BLKS="1 4 16 64 256 1024"
TPBS="32 64 128 256 512 1024"
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
    for (( i = 0; i < REPS; i++ )); do
	for blk in $BLKS; do
	    for tpb in $TPBS; do
		id="n${N}-d${D}"
		ifile="../input/${id}.txt"
		ofile="../output/${id}-b${blk}-t${tpb}-${prog}.data"
		echo $id - $blk - $tpb - $prog - $i
		cd workspace
		./$prog -k $K -d $D -i $ifile -m $M -t $T -s $S -c -blk $blk -tpb $tpb >> $ofile
		cd ..
	    done
	done
    done
done
