#!/bin/bash
#SBATCH --account=k1285
#SBATCH --job-name=lubm10240-16-0016-0256
#SBATCH --output=/project/k1285/CombBLAS/run/lubm10240/cores16-per-node/lubm10240_nodes0016_cores0256.log
#SBATCH --error=/project/k1285/CombBLAS/run/lubm10240/cores16-per-node/lubm10240_nodes0016_cores0256.log

srun --time=01:00:00 --hint=nomultithread --ntasks-per-node=16 --ntasks-per-socket=8 --nodes=16 --exclusive /project/k1285/CombBLAS/build/selfTests/lubm10240
