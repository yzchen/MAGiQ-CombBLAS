#!/bin/bash
#SBATCH --account=k1285
#SBATCH --job-name=get-dcsc-local-1
#SBATCH --output=/project/k1285/CombBLAS/run/dcsc/get-dcsc-local-lubm10240-nodes4-cores16.log
#SBATCH --error=/project/k1285/CombBLAS/run/err.log
#SBATCH --time=01:00:00
#SBATCH --threads-per-core=1
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive

srun /project/k1285/CombBLAS/build/selfTests/getdcsc /project/k1285/fuad/data/lubm10240/encoded.mm
