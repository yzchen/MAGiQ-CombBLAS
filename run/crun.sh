#!/bin/bash

# make project
mkdir -p ../build
cd ../build && cmake .. && make

# submit a job on shaheen
cd ../run && sbatch ./100k-nodes0512-cores4096.sh
