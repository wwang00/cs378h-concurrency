#!/bin/bash

rm -f /tmp/coordinator.log
rm -f /tmp/participant_*.log

./target/debug/cs380p-2pc -s .5 -c 2 -p 2 -r 10 -v 5 -m run
