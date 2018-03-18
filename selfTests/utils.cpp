#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
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

//template<class NT>
//bool set_element(PSpMat<NT>::MPI_DCCols &M, int i, int j, NT v) {
//    FullDistVec<int, int> ri(1, 0);
//    tmp.SetElement(0, i);
//    FullDistVec<int, int> ci(1, 0);
//    tmp.SetElement(0, j);
//    FullDistVec<int, NT> vi(1, 0);
//    tmp.SetElement(0, v);
//
//    M.SpAsgn(ri, ci, PSpMat<NT>::MPI_DCCols(ri, ci, vi));
//    return true;
//}
//
//template<class NT, class VT>
//PSpMat<NT>::MPI_DCCols mmul_scalar(const PSpMat<NT>::MPI_DCCols &M, VT s) {
//
//}
//
//FullyDistVec<int, ElementType> diagonalize(const PSpMat<ElementType>::MPI_DCCols &M) {
//    shared_ptr<CommGrid> fullWorld;
//    fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
//
//    FullyDistVec< int, ElementType> diag(fullWorld);
//
//    M.Reduce(diag, Row, std::bit_or<ElementType>() , 0);
//
//    return diag;
//}



void transpose(PSpMat<ElementType>::MPI_DCCols &M) {
   M.Transpose();
}

int main(int argc, char *argv[]) {
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 2) {
        if (myrank == 0) {
            cout << "Usage: ./utils <Matrix>" << endl;
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

        PSpMat<ElementType>::MPI_DCCols A;

        A.ReadDistribute(Mname, 0);
        A.PrintInfo();

        FullyDistVec<int, ElementType> rowsums(A.getcommgrid());
        FullyDistVec< int, ElementType> diag(fullWorld);

        A.Reduce(diag, Row, std::plus<ElementType>() , 0);

//        transpose(A);
//        A.PrintInfo();

    }

    MPI_Finalize();
    return 0;
}

