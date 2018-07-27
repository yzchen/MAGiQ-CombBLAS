#!/bin/bash
#SBATCH --account=k1285
#SBATCH --job-name=psort-test
#SBATCH --output=/scratch/cheny0l/CombBLAS/pmergesort/test_on_shaheen/run.out
#SBATCH --error=/scratch/cheny0l/CombBLAS/pmergesort/test_on_shaheen/run.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
##SBATCH --ntasks-per-socket=8
#SBATCH --exclusive

srun --cpu_bind=threads  /scratch/cheny0l/CombBLAS/pmergesort/test_pmergesort
