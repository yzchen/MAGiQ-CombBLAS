#!/bin/bash

rm -rf time.csv
touch time.csv
echo "read,transpose,diagonalize,mmul_scalar,mult1,mult2" > time.csv

./selfTests/timing
mpirun -np 4 ./selfTests/timing
mpirun -np 9 ./selfTests/timing
mpirun -np 16 ./selfTests/timing
mpirun -np 25 ./selfTests/timing
mpirun -np 36 ./selfTests/timing
mpirun -np 49 ./selfTests/timing
mpirun -np 64 ./selfTests/timing
mpirun -np 81 ./selfTests/timing
mpirun -np 100 ./selfTests/timing
