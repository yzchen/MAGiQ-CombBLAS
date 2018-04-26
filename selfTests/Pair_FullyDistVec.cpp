#include <iostream>
#include <functional>
#include <algorithm>
#include <sstream>
#include "../include/CombBLAS.h"

using namespace std;
using namespace combblas;

#define ElementType int
#define IndexType int

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

        FullyDistVec<IndexType, std::pair<ElementType, ElementType> > A(100, std::pair<ElementType, ElementType>(0, 0));
        A.SetElement(15, std::pair<ElementType, ElementType>(5, 5));

        int localsize = A.LocArrSize();
        auto v = A[15];
        if (myrank == 0) {
            cout << "local array size : " << localsize << endl;
            cout << "local value of index 15 : " << v.first << ", " << v.second << endl;
        }
    }

    MPI_Finalize();
    return 0;

}
