#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "../include/CombBLAS.h"
#include "../include/FullyDistSpVec.h"
#include "../include/FullyDistVec.h"

using namespace std;
using namespace combblas;

template<class NT>
class PSpMat {
public:
    typedef SpDCCols<int, NT> DCCols;
    typedef SpParMat<int, NT, DCCols> MPI_DCCols;
};

#define ElementType int

bool isTwo(ElementType t) {
    return t == 2;
}

int main(int argc, char *argv[]) {
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 1) {
        if (myrank == 0) {
            cout << "Usage: ./prune_mat" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    {
        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));

        MPI_Barrier(MPI_COMM_WORLD);

        int ris[3] = {1, 2, 3};
        std::vector<int> riv(ris, ris + sizeof(ris) / sizeof(int));

        int cis[3] = {3, 2, 1};
        std::vector<int> civ(cis, cis + sizeof(cis) / sizeof(int));

        int vis[3] = {1, 2, 3};
        std::vector<int> viv(vis, vis + sizeof(vis) / sizeof(int));

        FullyDistVec<int, ElementType> ri(riv, fullWorld);
        if (myrank == 0) {
            std::cout << "ri : ";
        }
        ri.DebugPrint();
        FullyDistVec<int, ElementType> ci(civ, fullWorld);
        if (myrank == 0) {
            std::cout << "ci : ";
        }
        ci.DebugPrint();
        FullyDistVec<int, ElementType> vi(viv, fullWorld);
        if (myrank == 0) {
            std::cout << "vi : ";
        }
        vi.DebugPrint();

        PSpMat<ElementType>::MPI_DCCols A(3, 3, ri, ci, vi);

        A.PrintInfo();

        A.Prune(isTwo);

        if (myrank == 0) {
            cout << "After prune 2 : " << endl;
        }
        A.PrintInfo();
    }

    MPI_Finalize();
    return 0;
}