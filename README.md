# MAGiQ(CombBLAS)

## Running on Shaheen

```
module swap PrgEnv-cray/6.0.4 PrgEnv-gnu
module load cray-mpich
```

example job script on Shaheen:

```
#!/bin/bash
#SBATCH --account=k1210
#SBATCH --job-name=512B-N2k-C2
#SBATCH --output=/CombBLAS/magiq_run/logs/trill/lubm512B-N2k-C2.log
#SBATCH --error=/CombBLAS/magiq_run/logs/trill/lubm512B-N2k-C2.err
#SBATCH --time=02:24:00
#SBATCH --nodes=2048
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive

module swap PrgEnv-cray/6.0.4 PrgEnv-gnu
module load cray-mpich

srun --cpu_bind=threads /CombBLAS/build/magiq_src/magiqScal /trill_exp/data_striped_144/lubm512B/paracoder_lubm512B.nt
```

run on local machine:

```
mpiexec -n 16 /CombBLAS/build/magiq_src/magiqScal /trill_exp/data_striped_144/lubm512B/paracoder_lubm512B.nt
```

## Project Structure

### magiq_src

contains src file for magiq(combBLAS) implementation

### magiq_include

contains header file for magiq(combBLAS) implementation

### external

all dependencies

### run

script about how to build and run

### build

not shown on github, all things are compiled here
