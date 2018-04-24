#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>
#include "../include/CombBLAS.h"

using namespace std;
using namespace combblas;

template<class NT>
class PSpMat {
public:
    typedef SpDCCols<int64_t, NT> DCCols;
    typedef SpParMat<int64_t, NT, DCCols> MPI_DCCols;
};

#define ElementType int

void mmul_scalar(PSpMat<ElementType>::MPI_DCCols &M, ElementType s) {
    M.Apply(bind2nd(multiplies<ElementType>(), s));
}

PSpMat<ElementType>::MPI_DCCols transpose(PSpMat<ElementType>::MPI_DCCols &M) {
    PSpMat<ElementType>::MPI_DCCols N(M);
    N.Transpose();
    return N;
}

PSpMat<ElementType>::MPI_DCCols diagonalize(const PSpMat<ElementType>::MPI_DCCols &M) {
    int dim = M.getnrow();

    FullyDistVec< int, ElementType> diag(M.getcommgrid());

    M.Reduce(diag, Row, std::logical_or<ElementType>() , 0);

    FullyDistVec<int, int> *rvec = new FullyDistVec<int, int>(diag.commGrid);
    rvec->iota(dim, 0);
    FullyDistVec<int, int> *qvec = new FullyDistVec<int, int>(diag.commGrid);
    qvec->iota(dim, 0);
    PSpMat<ElementType>::MPI_DCCols D(dim, dim, *rvec, *qvec, diag);

    return D;
}

int main(int argc, char *argv[]) {
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 1) {
        if (myrank == 0) {
            cout << "Usage: ./timing" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    {
        string Aname("/home/cheny0l/work/db245/fuad/data/lubm320/encoded.mm");
        ofstream ofs ("time.csv", std::ofstream::app);

//        ofs << "read,transpose,diagonalize,mmul_scalar,mult1,mult2\n";

        MPI_Barrier(MPI_COMM_WORLD);

        typedef PlusTimesSRing<ElementType, ElementType> PTINTINT;
        PSpMat<ElementType>::MPI_DCCols A(MPI_COMM_WORLD);

        double t1 = MPI_Wtime();
        A.ParallelReadMM(Aname, true, maximum<double>());
        A.PrintInfo();
        double t2 = MPI_Wtime();
        if (myrank == 0) {
            ofs << t2 - t1 << ",";
            cout << "read file takes : " << t2 - t1 << " s" << endl;
        }

        double t3 = MPI_Wtime();
        auto B = transpose(A);
        double t4 = MPI_Wtime();
        if(myrank == 0) {
            ofs << t4 - t3 << ",";
            cout << "transpose takes : " << t4 - t3 << " s" << endl;
        }

        double t5 = MPI_Wtime();
        auto C = diagonalize(A);
        double t6 = MPI_Wtime();
        if(myrank == 0) {
            ofs << t6 - t5 << ",";
            cout << "diagonalize takes : " << t6 - t5 << " s" << endl;
        }

        double t7 = MPI_Wtime();
        mmul_scalar(A, 2);
        double t8 = MPI_Wtime();
        if(myrank == 0) {
            ofs << t8 - t7 << ",";
            cout << "mmul_scalar takes : " << t8 - t7 << " s" << endl;
        }

        auto D(A);
        double t9 = MPI_Wtime();
        PSpMat<ElementType>::MPI_DCCols E = Mult_AnXBn_DoubleBuff<PTINTINT, ElementType, PSpMat<ElementType>::DCCols>(A, D);
        double t10 = MPI_Wtime();
        float im1 = E.LoadImbalance();
        if(myrank == 0) {
            ofs << t10 - t9 << ",";
            cout << "multiplication1 imbalance : " << im1 << endl;
            cout << "multiplication1 takes : " << t10 - t9 << " s" << endl;
        }

        double t11 = MPI_Wtime();
        PSpMat<ElementType>::MPI_DCCols F = PSpGEMM<PTINTINT>(A, D);
        double t12 = MPI_Wtime();
        float im2 = F.LoadImbalance();
        if(myrank == 0) {
            ofs << t12 - t11 << ",";
            cout << "multiplication2 imbalance : " << im2 << endl;
            cout << "multiplication2 takes : " << t12 - t11 << " s" << endl;
        }

        double t13 = MPI_Wtime();
        PSpMat<ElementType>::MPI_DCCols G = Mult_AnXBn_Synch<PTINTINT, ElementType, PSpMat<ElementType>::DCCols>(A, D);
        double t14 = MPI_Wtime();
        float im3 = G.LoadImbalance();
        if(myrank == 0) {
            ofs << t14 - t13 << ",";
            cout << "multiplication3 imbalance : " << im3 << endl;
            cout << "multiplication3 takes : " << t14 - t13 << " s" << endl;
        }

        auto H(A);
        double t15 = MPI_Wtime();
        H.Square<PTINTINT>();
        double t16 = MPI_Wtime();
        float im4 = H.LoadImbalance();
        if(myrank == 0) {
            ofs << t16 - t15 << "\n";
            cout << "multiplication4 imbalance : " << im4 << endl;
            cout << "multiplication4 takes : " << t16 - t15 << " s" << endl;
        }

        ofs.close();

    }

    MPI_Finalize();
    return 0;
}
