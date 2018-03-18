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

int main(int argc, char *argv[]) {
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 1) {
        if (myrank == 0) {
            cout << "Usage: ./constructmat" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    {
        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

        MPI_Barrier(MPI_COMM_WORLD);

        std::vector<int> riv(3, 5);
        std::vector<int> civ(3, 2);

        int vis[3] = {1, 2, 3};
        std::vector<int> viv(vis, vis + sizeof(vis) / sizeof(int));

        FullyDistVec<int, ElementType> ri(riv, fullWorld);
        std::cout << "ri : ";
        ri.DebugPrint();
        FullyDistVec<int, ElementType> ci(civ, fullWorld);
        std::cout << "ci : ";
        ci.DebugPrint();
        FullyDistVec<int, ElementType> vi(viv, fullWorld);
        std::cout << "vi : ";
        vi.DebugPrint();

        PSpMat<ElementType>::MPI_DCCols A(8, 8, ri, ci, vi);

        std::cout << "\nA is constructed with sumDuplicate = false" << std::endl;
        A.PrintInfo();

        FullyDistVec<int, ElementType> rowsumsA(fullWorld);

        A.Reduce(rowsumsA, Row, std::plus<ElementType>() , 0);

        std::cout << "A rowsums : ";
        rowsumsA.DebugPrint();

        PSpMat<ElementType>::MPI_DCCols B(8, 8, ri, ci, vi, true);

        std::cout << "\nB is constructed with sumDuplicate = true" << std::endl;
        B.PrintInfo();

        FullyDistVec<int, ElementType> rowsumsB(fullWorld);

        A.Reduce(rowsumsB, Row, std::plus<ElementType>() , 0);

        std::cout << "B rowsums : ";
        rowsumsB.DebugPrint();
    }

    MPI_Finalize();
    return 0;
}
