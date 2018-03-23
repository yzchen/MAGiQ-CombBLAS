#include <iostream>
#include <functional>
#include <algorithm>
#include <sstream>
#include "../include/CombBLAS.h"

using namespace std;
using namespace combblas;

template<class NT>
class PSpMat {
public:
    typedef SpDCCols<int, NT> DCCols;
    typedef SpParMat<int, NT, DCCols> MPI_DCCols;
};

#define ElementType int

int main(int argc, char *argv[]) {
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 3) {
        if (myrank == 0) {
            cout << "Usage: ./ReduceAdd <Matrix> <OutputPath>" << endl;
            cout << "<Matrix> is file address, and file should be in triples format, <OutputPath> is the vector output path" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    {
        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

        string Mname(argv[1]);
        string Vname(argv[2]);

        MPI_Barrier(MPI_COMM_WORLD);

        PSpMat<ElementType>::MPI_DCCols A(MPI_COMM_WORLD);

        double t1 = MPI_Wtime();
        A.ParallelReadMM(Mname, true, std::plus<ElementType>());
        double t2 = MPI_Wtime();
        if(myrank == 0) {
            cout << "read file takes " << t2 - t1 << " s" << endl;
        }
        A.PrintInfo();

        FullyDistVec<int, ElementType> rowsums(fullWorld);

        // 0 used for append when resizing vector, usually useless
        double t3 = MPI_Wtime();
        A.Reduce(rowsums, Row, std::plus<ElementType>() , 0);
        double t4 = MPI_Wtime();
        if(myrank == 0) {
            cout << "reduce-add takes " << t4 - t3 << " s" << endl;
        }

        rowsums.ParallelWrite(Vname, 1);
    }

    MPI_Finalize();
    return 0;
}
