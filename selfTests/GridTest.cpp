#include <iostream>
#include <functional>
#include <algorithm>
#include <sstream>
#include "../include/CombBLAS.h"

#define IndexType int
#define ElementType int

using namespace std;
using namespace combblas;

class PSpMat {
public:
    typedef SpDCCols<IndexType, ElementType> DCCols;
    typedef SpParMat<IndexType, ElementType, DCCols> MPI_DCCols;
};

int main(int argc, char *argv[]) {
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 2) {
        if (myrank == 0) {
            cout << "Usage: ./GridTest <MatrixA>" << endl;
            cout << "<MatrixA> is file addresses, and file should be in triples format" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    {
        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

        string Aname(argv[1]);

        MPI_Barrier(MPI_COMM_WORLD);

        PSpMat::MPI_DCCols A(MPI_COMM_WORLD);

        double t1 = MPI_Wtime();
        A.ReadDistribute(Aname, 0);
//        A.ParallelReadMM(Mname, true, maximum<ElementType>());
        double t2 = MPI_Wtime();
        if(myrank == 0) {
            cout << "read file takes " << t2 - t1 << " s" << endl;
        }
        A.PrintInfo();

        auto commGrid = A.getcommgrid();
        int grcols = commGrid->GetGridCols();
        int grrows = commGrid->GetGridRows();

        int myprocrow = commGrid->GetRankInProcRow();
        int myproccol = commGrid->GetRankInProcCol();
        int myr = commGrid->GetRank();

        // grrows == grcols == sqrt(#processes)
        cout << myrank << ", grcols = " << grcols << "  grrows = " << grrows << endl;

        // (myprocrow, myproccol) is the position of current process, myrank is same as mpi rank
        cout << myrank << ", myproccol = " << myproccol << "  myprocrow = " << myprocrow << "   myrank = " << myr << endl;
    }

    MPI_Finalize();
    return 0;
}
