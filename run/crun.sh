#!/bin/bash


cd ../build && make

cd ../run && sbatch ./100k-nodes0512-cores4096.sh
