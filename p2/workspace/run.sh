#!/bin/bash

go run *.go -input="simple.txt" -hash-workers=17 -data-workers=1 -comp-workers=1
