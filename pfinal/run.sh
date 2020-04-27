#!/bin/bash

IFILE=$PWD/input/input.txt

cd workspace
make all

./kalman_seq -i $IFILE

cd ..
