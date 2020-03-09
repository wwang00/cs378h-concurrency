#!/bin/bash

TESTS="simple coarse fine"
REPS=20

cd workspace
go build -o execute
cd ..

rm output/*

for input in $TESTS; do
    fin="$PWD/input/${input}.txt"
    # compute hashes
    for hw in 1 2 4 8 16; do
	fout="$PWD/output/${input}_hash_${hw}.txt"
	for (( i = 0; i < REPS; i++ )); do
	    echo hash $hw - $i
	    workspace/execute -d \
			      -input=$fin \
			      -hash-workers=$hw >> $fout
	done
    done
    # group hashes
    for hw in 1 2 4 8 16; do
	for (( dw = 1; dw <= hw; dw *= 2 )); do
	    fout="$PWD/output/${input}_group_${hw}_${dw}.txt"
	    for (( i = 0; i < REPS; i++ )); do
		echo group $hw $dw - $i
		workspace/execute -d \
				  -input=$fin \
				  -hash-workers=$hw \
				  -data-workers=$dw >> $fout 
	    done
	done
    done
    # compare trees
    for cw in 1 2 4 8 16; do
	fout="$PWD/output/${input}_comp_${cw}.txt"
	for (( i = 0; i < REPS; i++ )); do
	    echo comp $cw - $i
	    workspace/execute -d \
			      -input=$fin \
			      -hash-workers=1 \
			      -data-workers=1 \
			      -comp-workers=$cw >> $fout 
	done
    done
done

