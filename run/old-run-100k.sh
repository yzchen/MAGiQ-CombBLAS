#!/bin/bash
#SBATCH --account=k1285
#SBATCH --job-name=lubm100k-04-1024-4096
#SBATCH --output=/project/k1285/CombBLAS/run/lubm100k/lubm100k_nodes1024_cores4096.log
#SBATCH --error=/project/k1285/CombBLAS/run/err.log
#SBATCH --time=01:00:00
#SBATCH --threads-per-core=1
#SBATCH --nodes=1024
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive

srun /project/k1285/CombBLAS/build/selfTests/lubm100k
