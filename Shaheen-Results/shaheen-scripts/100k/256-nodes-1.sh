#!/bin/bash
#SBATCH --account=k1285
#SBATCH --job-name=100k-256-1
#SBATCH --output=/project/k1285/CombBLAS/run/lubm100k/lubm100k_nodes256-cores4k-1.log
#SBATCH --error=/project/k1285/CombBLAS/run/lubm100k/lubm100k_nodes256-cores4k-1.err
#SBATCH --nodes=256
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --ntasks-per-node=16
#SBATCH --ntasks-per-socket=8
#SBATCH --ntasks=4096
#SBATCH --exclusive

srun /project/k1285/CombBLAS/build/selfTests/lubm100k
