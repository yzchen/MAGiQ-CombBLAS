#!/bin/bash
#SBATCH --account=k1285 
#SBATCH --job-name=lubm10240_l2
#SBATCH --output=/project/k1285/CombBLAS/build/lubm10240_l2.log
#SBATCH --error=/project/k1285/CombBLAS/build/lubm10240_l2.err
#SBATCH --time=00:50:00
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=16

srun /project/k1285/CombBLAS/build/selfTests/lubm10240_l2
