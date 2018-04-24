#include <iostream>
#include <functional>
#include <algorithm>
#include <sstream>
#include "../include/CombBLAS.h"

using namespace std;
using namespace combblas;

#define ElementType int
#define IndexType int

template<class NT>
class PSpMat {
public:
    typedef SpDCCols<IndexType, NT> DCCols;
    typedef SpParMat<IndexType, NT, DCCols> MPI_DCCols;
};


int main(int argc, char *argv[]) {
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 1) {
        if (myrank == 0) {
            cout << "Usage: ./permute" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    {
        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));

        MPI_Barrier(MPI_COMM_WORLD);

        string Mname("/home/cheny0l/work/db245/CombBLAS_beta_16_1/TESTDATA/gen_10_10_25.txt");
        PSpMat<IndexType>::MPI_DCCols G(MPI_COMM_WORLD);

        G.ParallelReadMM(Mname, true, maximum<ElementType>());
        G.PrintInfo();

        int vis[3] = {1, 2, 3};
        std::vector<int> viv(vis, vis + sizeof(vis) / sizeof(int));
        FullyDistVec<IndexType, ElementType> vv(viv, fullWorld);

        G.DimApply(Row, vv, std::plus<ElementType>());
        G.SaveGathered("dim-apply.txt");
   }

    MPI_Finalize();
    return 0;
}
