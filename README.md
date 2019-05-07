# MAGiQ(CombBLAS)

[Combinatorial BLAS](https://people.eecs.berkeley.edu/~aydin/CombBLAS/html/index.html)(version 1.6.1, Jan 2018)

## Prerequisites

- MPI (see first a few lines in CMakeLists.txt)

- cmake >= 3.0

- C++11/C++14

## Compile && Run

#### Compile

1. normal compilation

    ```
    mkdir -p build
    cd ../build && cmake .. && make -j
    ```

2. use `COMBBLAS_DEBUG` mode to enable combblas debug information output:

    ```
    mkdir -p build
    cd ../build && cmake -D COMBBLAS_DEBUG=ON .. && make -j
    ```

3. use `MAGIQ_DEBUG` mode to enable magiq debug information output:

    ```
    mkdir -p build
    cd ../build && cmake -D MAGIQ_DEBUG=ON .. && make -j
    ```

4. build on Shaheen

    first load proper modules:
    ```
    module swap PrgEnv-cray/6.0.4 PrgEnv-gnu
    module load cray-mpich
    ```

    change the first two line of `CMakeLists.txt`, make following changes :

    ```
    -SET(CMAKE_CXX_COMPILER mpic++)
    -SET(CMAKE_C_COMPILER mpicc)
    +SET(CMAKE_CXX_COMPILER CC)
    +SET(CMAKE_C_COMPILER cc)
    ```

    then do as above three sections suggested, by setting compiler to `CC/cc` you can only run the executable file with `srun`.

#### Run

1. run(in `$MAGiQ_ROOT/build`) hard coded program:

    ```
    mpirun -np 16 ./magiq_src/magiqScal ../data/paracoder_lubm1B.nt
    ```

2. run(in `$MAGiQ_ROOT/build`) sparql parser program:

    ```
    mpirun -np 16 ./magiq_src/magiqParse ../data/paracoder_lubm1B.nt ../examples/queries/lubm/q7.txt 1
    ```

3. run on Shaheen

    example job script on Shaheen:

    ```
    #!/bin/bash
    #SBATCH --account=k1210
    #SBATCH --job-name=test
    #SBATCH --output=otest
    #SBATCH --error=etest
    #SBATCH --nodes=1
    #SBATCH --time=10:00
    #SBATCH --hint=nomultithread
    #SBATCH --ntasks-per-node=1
    #SBATCH --ntasks=1
    #SBATCH --exclusive

    module swap PrgEnv-cray/6.0.4 PrgEnv-gnu
    module load cray-mpich

    cd /project/k1285/CombBLAS/
    srun ./build/magiq_src/magiqParse ./examples/data/poster.nt ./examples/queries/poster/query.txt
    ```

## Project Structure

### docs

some documents about this project

### example

example graph data and query files

### external

all dependencies

### include 

CombBLAS header files

### magiq_include

contains header file for magiq(CombBLAS) implementation

### magiq_src

contains src file for magiq(CombBLAS) implementation

- magiqScal.cpp

    hard coded program for lubm queries

- magiqParse.cc

    automatic query translation and execution

### src

CombBLAS source files

### build

not shown on github, all things are compiled here
