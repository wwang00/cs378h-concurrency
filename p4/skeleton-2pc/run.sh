#!/bin/bash

if [ "$1" == "" ]; then
    MODE=run
    rm -f /tmp/coordinator.log
    rm -f /tmp/participant_*.log
else
    MODE=check
fi

./target/debug/cs380p-2pc -s .8 -c 5 -p 5 -r 10 -v 5 -m $MODE
