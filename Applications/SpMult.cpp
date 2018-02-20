#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "../CombBLAS.h"

using namespace std;
using namespace combblas;

template<class NT>
class PSpMat {
public:
    typedef SpDCCols<int, NT> DCCols;
    typedef SpParMat<int, NT, DCCols> MPI_DCCols;
};

#define ElementType double


int main(int argc, char *argv[]) {
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 3) {
        if (myrank == 0) {
            cout << "Usage: ./SpMult <MatrixA> <MatrixB>" << endl;
            cout << "<MatrixA>, <MatrixB> are file addresses, and files should be in triples format" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    {
        string Aname(argv[1]);
        string Bname(argv[2]);

        typedef PlusTimesSRing<ElementType, ElementType> PTDOUBLEDOUBLE;
        PSpMat<ElementType>::MPI_DCCols A, B;

        A.ReadDistribute(Aname, 0);
        A.PrintInfo();

        B.ReadDistribute(Bname, 0);
        B.PrintInfo();

        double t1 = MPI_Wtime();
        PSpMat<ElementType>::MPI_DCCols C = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE, ElementType, PSpMat<ElementType>::DCCols>(A, B);
        double t2 = MPI_Wtime();
        printf("multiplication takes %.6lf s\n", (t2 - t1));

        C.PrintInfo();
    }

    MPI_Finalize();
    return 0;
}

