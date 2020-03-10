#!/bin/bash

for f in $(echo output/*); do
    avg=$(python3 print_avgs.py $f)
    echo "$f - $avg"
done
