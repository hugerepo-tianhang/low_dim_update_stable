#!/bin/bash

pid=$(ps -opid= -C sandbox1.sh)
echo $pid
while [ -d /proc/$pid ] ; do
    echo "not uet"
    sleep 1
done && ./job_dev.sh