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

bool isZero(ElementType t) {
    return t == 0;
}

int main(int argc, char *argv[]) {
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 1) {
        if (myrank == 0) {
            cout << "Usage: ./SemiringMult" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    {
        string Aname("/home/cheny0l/work/db245/CombBLAS_beta_16_1/TESTDATA/gen_2_2_3.txt");
        string Bname("/home/cheny0l/work/db245/CombBLAS_beta_16_1/TESTDATA/gen_2_2_2.txt");

        MPI_Barrier(MPI_COMM_WORLD);

        typedef RDFRing<ElementType, ElementType> PTINTINT;
        PSpMat<ElementType>::MPI_DCCols A(MPI_COMM_WORLD), B(MPI_COMM_WORLD);

        A.ReadDistribute(Aname, 0);
        A.PrintInfo();

        B.ReadDistribute(Bname, 0);
        B.PrintInfo();

        double t1 = MPI_Wtime();
        PSpMat<ElementType>::MPI_DCCols C = Mult_AnXBn_DoubleBuff<PTINTINT, ElementType, PSpMat<ElementType>::DCCols>(A, B);
        double t2 = MPI_Wtime();
        if(myrank == 0) {
            cout << "multiplication takes " << t2 - t1 << " s" << endl;
        }

        C.PrintInfo();

        C.Prune(isZero);

        C.PrintInfo();

        C.SaveGathered("/home/cheny0l/work/db245/CombBLAS_beta_16_1/TESTDATA/test_rdf_semiring.del");
    }

    MPI_Finalize();
    return 0;
}
