#!/bin/bash

go run *.go -input="../input/coarse.txt" -hash-workers=$1 -data-workers=$2 -comp-workers=$3
