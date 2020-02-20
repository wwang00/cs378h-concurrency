#!/bin/bash

go run *.go -d -input="../input/fine.txt" -hash-workers=$1 -data-workers=$2 -comp-workers=$3
