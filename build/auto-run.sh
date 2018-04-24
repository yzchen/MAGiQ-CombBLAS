#!/bin/bash

./selfTests/lubm10240_l7 | tee lubm10240_l7_node1.log
mpirun -np 4 ./selfTests/lubm10240_l7 | tee lubm10240_l7_node4.log
mpirun -np 16 ./selfTests/lubm10240_l7 | tee lubm10240_l7_node16.log
