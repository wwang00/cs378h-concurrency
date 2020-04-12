#!/bin/bash

if [ "$1" == "" ]; then
    MODE=run
    rm -rf /tmp/coordinator.log
    rm -rf /tmp/participant_*.log
else
    MODE=check
fi

./target/debug/cs380p-2pc -f 0.1 -s .9 -S .99 -c 1 -p 4 -r 100 -v 5 -m $MODE
