#!/bin/bash
#SBATCH --account=k1285
#SBATCH --job-name=lubm10240-16-0004-0064
#SBATCH --output=/project/k1285/CombBLAS/run/lubm10240/cores16-per-node/lubm10240_nodes0004_cores0064.log
#SBATCH --error=/project/k1285/CombBLAS/run/lubm10240/cores16-per-node/lubm10240_nodes0004_cores0064.log

srun --time=01:00:00 --hint=nomultithread --ntasks-per-node=16 --ntasks-per-socket=8 --nodes=4 --exclusive /project/k1285/CombBLAS/build/selfTests/lubm10240
