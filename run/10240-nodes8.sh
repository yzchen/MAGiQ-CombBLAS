#!/bin/bash
#SBATCH --account=k1285
#SBATCH --job-name=lubm10240_l7
#SBATCH --output=/project/k1285/CombBLAS/run/lubm10240_nodes0008_cores0016.log
#SBATCH --error=/project/k1285/CombBLAS/run/err.log
#SBATCH --time=00:30:00
#SBATCH --threads-per-core=1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=2
#SBATCH --exclusive

srun /project/k1285/CombBLAS/build/selfTests/lubm10240
