#!/bin/bash
#SBATCH --account=k1285
#SBATCH --job-name=reduct-1
#SBATCH --output=/project/k1285/CombBLAS/run/reduct/reduct-lubm10240-nodes16-cores256.log
#SBATCH --error=/project/k1285/CombBLAS/run/err.log
#SBATCH --time=01:00:00
#SBATCH --threads-per-core=1
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=16
#SBATCH --exclusive

srun /project/k1285/CombBLAS/build/selfTests/reduct /project/k1285/fuad/data/lubm10240/encoded.mm
