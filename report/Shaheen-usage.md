## Shaheen

#### Login

Need password and a OATH, got it from https://www.hpc.kaust.edu.sa/user/login

Actually it doesn't matter you login from other ip addresses, just need OATH

#### Basic observations

Shaheen : SUSE Linux

`Linux cdl3 4.4.103-6.38-default #1 SMP Mon Dec 25 20:44:33 UTC 2017 (e4b9067) x86_64 x86_64 x86_64 GNU/Linux`

- what Shaheen has :

    * gcc ( 4.8.5 )

    * docker ( 17.06.0-ce )

- whst Shaheen doesn't have :

    * mpi compiler (mpicc)

You can write mpi program, but can not use `mpicc` / `mpirun` commands.

~~Need to use container to run the CombBLAS~~

#### How to do things without sudo in remote server

[Linuxbrew](http://linuxbrew.sh/) is a good solution for you

1. installation : execute following script

    `sh -c "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install.sh)"`

2. configuration : add this to `$HOME/.bashrc`

    `export PATH=$PATH:$HOME/.linuxbrew/bin`

3. test : install a simple library

    `brew install hello`

    `hello` --> output : Hello, World!

*Drawback : installation is slow on Shaheen*

4. usage : install openmpi

    `brew install openmpi`

    then you can build CombBLAS correctly and successfully run self tests

5. uninstall : I want to try use Shaheen mpi, not my local mpi

    `ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/uninstall)"`

    `rm -rf .linuxbrew/`

    and remove `linuxbrew/bin` in your system `PATH`

Although linuxbrew is very nice solution, but the problem is that it will install a lot libraries in home directory and many of them can be found in system directory.

#### Shaheen mpi test program

core code :

`printf("Hello world from processor %s, rank %d out of %d processors\n", processor_name, world_rank, world_size);`

submit.sh :

```
#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=res.txt
#SBATCH --partition=workq
#SBATCH --ntasks=4
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

srun hello.mpi
```

by setting `ntasks=4`, you will get a 4-cpus to run this mpi program.

#### Run CombBLAS test on Shaheen
