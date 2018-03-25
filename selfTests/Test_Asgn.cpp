#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "../include/CombBLAS.h"

using namespace std;
using namespace combblas;

int main(int argc, char* argv[])
{
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    if(argc < 6)
    {
        if(myrank == 0)
        {
            cout << "Usage: ./test_asgn <BASEADDRESS> <Matrix> <RHSMatrix> <VectorRowIndices> <VectorColIndices>" << endl;
            cout << "Example: ./test_asgn ../mfiles A_100x100.txt dense_20x30matrix.txt 20outta100.txt 30outta100.txt" << endl;
            cout << "Input files should be under <BASEADDRESS> in tuples format" << endl;
        }
        MPI_Finalize();
        return -1;
    }
    {
        string directory(argv[1]);

        string normalname(argv[2]);
        string rhsmatname(argv[3]);
        string vec1name(argv[4]);
        string vec2name(argv[5]);

        normalname = directory+"/"+normalname;
        rhsmatname = directory+"/"+rhsmatname;
        vec1name = directory+"/"+vec1name;
        vec2name = directory+"/"+vec2name;

        ifstream inputvec1(vec1name.c_str());
        ifstream inputvec2(vec2name.c_str());

        MPI_Barrier(MPI_COMM_WORLD);

        typedef SpParMat <int, double , SpDCCols<int, double> > PARDBMAT;

        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

        PARDBMAT A(fullWorld);
        PARDBMAT B(fullWorld);
        PARDBMAT C(fullWorld);
        FullyDistVec<int,int> vec1(fullWorld);
        FullyDistVec<int,int> vec2(fullWorld);

        A.ReadDistribute(normalname, 0);
        B.ReadDistribute(rhsmatname, 0);
        vec1.ReadDistribute(inputvec1, 0);
        vec2.ReadDistribute(inputvec2, 0);

        vec1.Apply(bind2nd(minus<int>(), 1));	// For 0-based indexing
        vec2.Apply(bind2nd(minus<int>(), 1));

//        A.PrintInfo();
        A.SpAsgn(vec1, vec2, B);
//        A.PrintInfo();
        A.SaveGathered("out_asgn.del");

        inputvec1.clear();
        inputvec1.close();
        inputvec2.clear();
        inputvec2.close();
    }

    MPI_Finalize();
    return 0;
}
