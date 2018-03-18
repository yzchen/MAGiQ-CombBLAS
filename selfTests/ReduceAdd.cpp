#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "../CombBLAS.h"
#include "../FullyDistSpVec.h"
#include "../FullyDistVec.h"

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

        PSpMat<ElementType>::MPI_DCCols A;

        A.ReadDistribute(Mname, 0);
        A.PrintInfo();

        FullyDistVec< int, ElementType> rowsums_control(fullWorld);

        A.Reduce(rowsums_control, Row, std::plus<ElementType>() , 0);

        rowsums_control.ParallelWrite(Vname, 1);
    }

    MPI_Finalize();
    return 0;
}