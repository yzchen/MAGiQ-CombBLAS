# MAGiQ(CombBLAS)

[Combinatorial BLAS](https://people.eecs.berkeley.edu/~aydin/CombBLAS/html/index.html)(version 1.6.1, Jan 2018)

## Prerequisites

- MPI (see first a few lines in CMakeLists.txt)

- cmake >= 3.0

- C++11/C++14

## Compile && Run

```
mkdir -p ../build
cd ../build && cmake .. && make
```

Or to use `COMBBLAS_DEBUG` mode to enable combblas debug information output:

```
mkdir -p ../build
cd ../build && cmake -D COMBBLAS_DEBUG=ON .. && make
```

Or to use `MAGIQ_DEBUG` mode to enable magiq debug information output:

```
mkdir -p ../build
cd ../build && cmake -D MAGIQ_DEBUG=ON .. && make
```

Run(in `$MAGiQ_ROOT/build`):

```
mpirun -np 16 ./magiq_src/magiqScal ../data/paracoder_lubm1B.nt
```

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

## Intermediate Program Parser

A very simple domain specific language for matrix algebra operations in this scenario is implemented.
However, the functionalities are limited because it's only a simple string parser.

Following shows lubm benchmark queries(Note that only this kind of language is accepted):

- L1

    ```
    m_5_0 = G ⊗ I^22638*6
    m_3_5 = G ⊗ m_5_0.D*10
    m_1_3 = G.T ⊗ m_3_5.D*6
    m_1_3 = I^24 × m_1_3
    m_4_3 = G.T ⊗ m_1_3.T.D*2
    m_2_4 = G.T ⊗ m_4_3.D*6
    m_2_4 = I^8622222 × m_2_4
    m_6_4 = G ⊗ m_2_4.T.D*11
    m_6_4 = m_3_5.T.D × m_6_4
    m_3_5 = m_3_5 × m_6_4.D
    m_4_3 = m_6_4.T.D × m_4_3
    m_3_5 = m_4_3.T.D × m_3_5
    m_5_0 = m_3_5.T.D × m_5_0
    ```

- L2

    ```
    m_1_0 = G ⊗ I^79*6
    m_2_1 = G.T ⊗ m_1_0.D*3
    m_1_0 = m_2_1.T.D × m_1_0
    ```

- L3

    ```
    m_5_0 = G ⊗ I^22638*6
    m_3_5 = G ⊗ m_5_0.D*10
    m_1_3 = G.T  ⊗ m_3_5.D*6
    m_1_3 = I^43 × m_1_3
    m_4_3 = G.T ⊗ m_1_3.T.D*2
    m_2_4 = G.T ⊗ m_4_3.D*6
    m_2_4 = I^8622222 × m_2_4
    m_6_4 = G ⊗ m_2_4.T.D*11
    m_6_4 = m_3_5.T.D × m_6_4
    m_3_5 = m_3_5 × m_6_4.D
    m_4_3 = m_6_4.T.D × m_4_3
    m_3_5 = m_4_3.T.D × m_3_5
    m_5_0 = m_3_5.T.D × m_5_0
    ```

- L4

    ```
    m_2_0 = G ⊗ I^11*5
    m_1_2 = G.T ⊗ m_2_0.D*6
    m_1_2 = I^1345 × m_1_2
    m_3_2 = G.T ⊗ m_1_2.T.D*3
    m_4_2 = G.T ⊗ m_3_2.T.D*12
    m_5_2 = G.T ⊗ m_4_2.T.D*9
    m_2_0 = m_5_2.T.D × m_2_0
    ```

- L5

    ```
    m_3_0 = G ⊗ I^1345*6
    m_4_3 = G.T ⊗ m_3_0.D*5
    m_1_4 = G.T ⊗ m_4_3.D*6
    m_1_4 = I^22638 × m_1_4
    m_2_4 = G.T ⊗ m_1_4.T.D*11
    m_2_4 = I^40169 × m_2_4
    m_4_3 = m_2_4.T.D × m_4_3
    m_3_0 = m_4_3.T.D × m_3_0
    ```

- L6

    ```
    m_2_0 = G ⊗ I^11*11
    m_1_2 = G.T ⊗ m_2_0.D*6
    m_1_2 = I^357 × m_1_2
    m_2_0 = m_1_2.T.D × m_2_0
    ```

- L7

    ```
    m_5_0 = G ⊗ I^1345*6
    m_3_5 = G ⊗ m_5_0.D*13
    m_1_3 = G.T ⊗ m_3_5.D*6
    m_1_3 = I^43 × m_1_3
    m_4_3 = G.T ⊗ m_1_3.T.D*8
    m_2_4 = G.T ⊗ m_4_3.D*6
    m_2_4 = 1^79 × m_2_4
    m_6_4 = G ⊗ m_2_4.T.D*4
    m_6_4 = m_3_5.T.D × m_6_4
    m_3_5 = m_3_5 × m_6_4.D
    m_4_3 = m_6_4.T.D × m_4_3
    m_3_5 = m_4_3.T.D × m_3_5
    m_5_0 = m_3_5.T.D × m_5_0
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

### data 

not shown on github, currently hard coded magiqScal code is degined for `paracoder_lubmXXXB.nt` datasets.