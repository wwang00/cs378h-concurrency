#!/bin/bash

if [ "$1" == "" ]; then
    MODE=run
    rm -rf ~/tmp/coordinator.log
    rm -rf ~/tmp/participant_*.log
else
    MODE=check
fi

./target/debug/cs380p-2pc -f 0.02 -s .95 -S .99 -c 1 -p 8 -r 50 -v 5 -m $MODE

if [ "$MODE" == "run" ]; then
    ./run.sh c
fi
