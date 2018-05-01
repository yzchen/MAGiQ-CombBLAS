#!/bin/bash
#SBATCH --account=k1285
#SBATCH --job-name=100k-s256-1
#SBATCH --output=/project/k1285/CombBLAS/run/lubm100k/lubm100k_nodes256-cores1k-1.log
#SBATCH --error=/project/k1285/CombBLAS/run/lubm100k/lubm100k_nodes256-cores1k-1.err
#SBATCH --nodes=256
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --ntasks=1024
#SBATCH --exclusive

srun /project/k1285/CombBLAS/build/selfTests/lubm100k
