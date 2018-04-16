#!/bin/bash

rm -f time.csv
touch time.csv
echo "read,transpose,diagonalize,mmul_scalar,mult1,mult2,mult3,mult4" > time.csv

./selfTests/timing
mpirun -np 4 ./selfTests/timing
mpirun -np 9 ./selfTests/timing
mpirun -np 16 ./selfTests/timing
