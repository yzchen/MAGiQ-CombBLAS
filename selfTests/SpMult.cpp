#include <mpi.h>
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

        MPI_Barrier(MPI_COMM_WORLD);

        typedef PlusTimesSRing<ElementType, ElementType> PTINTINT;
        PSpMat<ElementType>::MPI_DCCols A(MPI_COMM_WORLD), B(MPI_COMM_WORLD);

        double t1 = MPI_Wtime();

//        A.ReadDistribute(Aname, 0);
        A.ParallelReadMM(Aname, true, maximum<double>());
        A.PrintInfo();

        double t2 = MPI_Wtime();

        if (myrank == 0) {
            cout << "read file takes : " << (t2 - t1) << " s" << endl;
        }

        double t3 = MPI_Wtime();

//        B.ReadDistribute(Bname, 0);
        B.ParallelReadMM(Bname, true, maximum<double>());
        B.PrintInfo();

        double t4 = MPI_Wtime();

        if (myrank == 0) {
            cout << "read file takes : " << (t4 - t3) << " s" << endl;
        }

        double t5 = MPI_Wtime();
        PSpMat<ElementType>::MPI_DCCols C = Mult_AnXBn_DoubleBuff<PTINTINT, ElementType, PSpMat<ElementType>::DCCols>(A, B);
        MPI_Barrier(MPI_COMM_WORLD);
        double t6 = MPI_Wtime();
        if(myrank == 0) {
            cout << "multiplication takes " << t6 - t5 << " s" << endl;
        }

        double t7 = MPI_Wtime();
        PSpMat<ElementType>::MPI_DCCols D = PSpGEMM<PTINTINT>(A,B);
        MPI_Barrier(MPI_COMM_WORLD);
        double t8 = MPI_Wtime();
        if(myrank == 0) {
            cout << "multiplication takes " << t8 - t7 << " s" << endl;
        }

//        double t9 = MPI_Wtime();
//        PSpMat<ElementType>::MPI_DCCols E = SpGEMM<PTINTINT>(A,B,2,1,1,1,1,1,1);
//        MPI_Barrier(MPI_COMM_WORLD);
//        double t10 = MPI_Wtime();
//        if(myrank == 0) {
//            cout << "multiplication takes " << t10 - t9 << " s" << endl;
//        }

    }

    MPI_Finalize();
    return 0;
}
