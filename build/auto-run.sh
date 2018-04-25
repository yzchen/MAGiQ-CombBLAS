#!/bin/bash

./selfTests/lubm10240_l7 1 | tee logs-2018-04-25/lubm10240_l7_node1_dim.log
mpirun -np 4 ./selfTests/lubm10240_l7 1 | tee logs-2018-04-25/lubm10240_l7_node4_dim.log
mpirun -np 16 ./selfTests/lubm10240_l7 1 | tee logs-2018-04-25/lubm10240_l7_node16_dim.log
