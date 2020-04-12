#!/bin/bash

if [ "$1" == "" ]; then
    MODE=run
    rm -rf /tmp/coordinator.log
    rm -rf /tmp/participant_*.log
else
    MODE=check
fi

./target/debug/cs380p-2pc -s .9 -S .95 -c 8 -p 8 -r 10 -v 5 -m $MODE
