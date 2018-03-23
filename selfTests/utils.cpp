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

void set_element(PSpMat<ElementType>::MPI_DCCols &M, int i, int j, ElementType v) {

    std::vector<int> riv(1, i);
    std::vector<int> civ(1, j);
    std::vector<ElementType> viv(1, v);

    FullyDistVec<int, ElementType> ri(riv, M.getcommgrid());
    FullyDistVec<int, ElementType> ci(civ, M.getcommgrid());
    FullyDistVec<int, ElementType> vi(viv, M.getcommgrid());

    ri.Apply(bind2nd(minus<int>(), 1));
    ci.Apply(bind2nd(minus<int>(), 1));

    PSpMat<ElementType>::MPI_DCCols B(M.getnrow(), M.getncol(), ri, ci, vi);
//    std::cout << "B : ";
//    B.PrintInfo();

    M.Prune(ri, ci);
    M += B;
}

void mmul_scalar(PSpMat<ElementType>::MPI_DCCols &M, ElementType s) {
    M.Apply(bind2nd(multiplies<ElementType>(), s));
}

PSpMat<ElementType>::MPI_DCCols diagonalize(const PSpMat<ElementType>::MPI_DCCols &M) {
    int dim = M.getnrow();

    FullyDistVec< int, ElementType> diag(M.getcommgrid());

    M.Reduce(diag, Row, std::logical_or<ElementType>() , 0);

    FullyDistVec<int, int> *rvec = new FullyDistVec<int, int>(diag.commGrid);
    FullyDistVec<int, int> *qvec = new FullyDistVec<int, int>(diag.commGrid);
    PSpMat<ElementType>::MPI_DCCols D(dim, dim, *rvec, *qvec, 0);

    for (int i = 1; i <= dim; ++i) {
        set_element(D, i, i, diag.GetElement(i - 1));
    }

    return D;
}

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

        PSpMat<ElementType>::MPI_DCCols A(MPI_COMM_WORLD);

        A.ReadDistribute(Mname, 0);
        if(myrank == 0) {
            std::cout << "\noriginal A : ";
            std::cout<<endl;
        }
        A.PrintInfo();

        // 1. test set element
        PSpMat<ElementType>::MPI_DCCols A1 = A;
        set_element(A1, 8, 5, 100);
        if(myrank == 0) {
            std::cout << "\nafter set element : ";
            std::cout<<endl;
        }
        A1.PrintInfo();
        A1.SaveGathered("set_element_A.out");

        // 2. test multiplying a scalar
        PSpMat<ElementType>::MPI_DCCols A2 = A;
        mmul_scalar(A2, 2);
        if(myrank == 0) {
            std::cout << "\nafter multiply scalar : ";
            std::cout<<endl;
        }
        A2.PrintInfo();
        A2.SaveGathered("mmul_scalar_A.out");

        // 3. test diagonalize
        PSpMat<ElementType>::MPI_DCCols A3 = A;
        PSpMat<ElementType>::MPI_DCCols D = diagonalize(A3);
        if(myrank == 0) {
            std::cout << "\nafter diagonalize : ";
            std::cout<<endl;
        }
        D.PrintInfo();
        D.SaveGathered("diagonalize_A.out");

        // 4. test transpose
        PSpMat<ElementType>::MPI_DCCols A4 = A;
        transpose(A4);
        if(myrank == 0) {
            std::cout << "\nafter transpose : ";
            std::cout<<endl;
        }
        A4.PrintInfo();
        A4.SaveGathered("transpose_A.out");

    }

    MPI_Finalize();
    return 0;
}

