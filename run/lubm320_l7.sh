#!/bin/bash
#
#SBATCH --job-name=lubm320_l7
#SBATCH --output=res_lubm320_l7.txt
#SBATCH --partition=workq
#SBATCH --ntasks=16
#SBATCH --time=00:01:00

srun /project/k1285/CombBLAS/build/selfTests/lubm320_l7
