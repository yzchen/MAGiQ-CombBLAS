#!/bin/bash


cd ../build && make

cd ../run && sbatch ./ll6-resgen.sh
