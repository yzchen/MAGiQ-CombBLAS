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

    if (argc < 2) {
        if (myrank == 0) {
            cout << "Usage: ./SpRefTest <Matrix>" << endl;
            cout << "<Matrix> is file address, and file should be in triples format" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    {
        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

        string Mname(argv[1]);

        MPI_Barrier(MPI_COMM_WORLD);

        PSpMat<ElementType>::MPI_DCCols A(MPI_COMM_WORLD);

        double t1 = MPI_Wtime();
        A.ReadDistribute(Mname, 0);
//        A.ParallelReadMM(Mname, true, maximum<ElementType>());
        double t2 = MPI_Wtime();
        if(myrank == 0) {
            cout << "read file takes " << t2 - t1 << " s" << endl;
        }
        A.PrintInfo();

        std::vector<int> Ai;
        Ai.push_back(67);
        auto Ac = A.SubsRefCol(Ai);

        Ac.PrintInfo();
    }

    MPI_Finalize();
    return 0;
}
