#!/bin/bash

go run *.go -input="$1" -hash-workers=1 -data-workers=1 -comp-workers=1
