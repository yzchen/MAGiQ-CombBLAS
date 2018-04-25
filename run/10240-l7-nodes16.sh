#!/bin/bash
#SBATCH --account=k1285
#SBATCH --job-name=lubm10240_l7
#SBATCH --output=/project/k1285/CombBLAS/run/lubm10240_l7_nodes16.log
#SBATCH --error=/project/k1285/CombBLAS/run/err.log
#SBATCH --time=00:30:00
#SBATCH --threads-per-core=1
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=16
#SBATCH --exclusive

srun /project/k1285/CombBLAS/build/selfTests/lubm10240_l7 0
