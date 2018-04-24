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
    typedef SpDCCols<int64_t, NT> DCCols;
    typedef SpParMat<int64_t, NT, DCCols> MPI_DCCols;
};

#define ElementType int

typedef RDFRing<ElementType, ElementType> RDFINTINT;
typedef PlusTimesSRing<ElementType, ElementType> PTINTINT;

void mmul_scalar(PSpMat<ElementType>::MPI_DCCols &M, ElementType s) {
    M.Apply(bind2nd(multiplies<ElementType>(), s));
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

PSpMat<ElementType>::MPI_DCCols transpose(PSpMat<ElementType>::MPI_DCCols &M) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t1 = MPI_Wtime();
    PSpMat<ElementType>::MPI_DCCols N(M);
    double t2 = MPI_Wtime();
    if (myrank == 0) {
        cout << "       trans 1st step takes " << (t2 - t1) << " s" << endl;
    }

    double t3 = MPI_Wtime();
    N.Transpose();
    double t4 = MPI_Wtime();
    if (myrank == 0) {
        cout << "       trans 2nd step takes " << (t4 - t3) << " s" << endl;
    }

    return N;
}

bool isZero(ElementType t) {
    return t == 0;
}

bool isNotZero(ElementType t) {
    return t != 0;
}

static double total_mult_time = 0.0;

void printReducedInfo(PSpMat<ElementType>::MPI_DCCols &M){
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t1 = MPI_Wtime();

    int nnz1 = M.getnnz();

    FullyDistVec<int, ElementType> rowsums1(M.getcommgrid());
    M.Reduce(rowsums1, Row, std::plus<ElementType>() , 0);
    FullyDistVec<int, ElementType> colsums1(M.getcommgrid());
    M.Reduce(colsums1, Column, std::plus<ElementType>() , 0);
    int nnzrows1 = rowsums1.Count(isNotZero);
    int nnzcols1 = colsums1.Count(isNotZero);

    double t2 = MPI_Wtime();
    if (myrank == 0) {
        cout << "    enum takes " << (t2 - t1) << " s" << endl;
    }

    if (myrank == 0) {
        cout << nnz1 << " [ " << nnzrows1 << ", " << nnzcols1 << " ]\n" << endl;
    }
}

template  <typename  SR>
void multPrune(PSpMat<ElementType>::MPI_DCCols &A, PSpMat<ElementType>::MPI_DCCols &B, PSpMat<ElementType>::MPI_DCCols &C, bool clearA = false, bool clearB = false) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t1 = MPI_Wtime();
    C = Mult_AnXBn_DoubleBuff<SR, ElementType, PSpMat<ElementType>::DCCols>(A, B, clearA, clearB);
//    C = PSpGEMM<SR>(A, B, clearA, clearB);
    double t2 = MPI_Wtime();

    if (myrank == 0) {
        total_mult_time += (t2 - t1);
        cout << "    multiplication takes: " << (t2 - t1) << " s" << endl;
    }

    double t3 = MPI_Wtime();
    C.Prune(isZero);
    double t4 = MPI_Wtime();

    if (myrank == 0) {
        cout << "    prune takes: " << (t4 - t3) << " s" << endl;
    }

    printReducedInfo(C);
}

void lubm320_L2(PSpMat<ElementType>::MPI_DCCols &G) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t1_trans = MPI_Wtime();
    auto tG = transpose(G);
    double t2_trans = MPI_Wtime();
    if (myrank == 0) {
        cout << "    transpose G takes : " << (t2_trans - t1_trans) << " s\n" <<endl;
    }

    // start count time
    double t1 = MPI_Wtime();

    int nrow = G.getnrow(), ncol = G.getncol();
    std::vector<int> riv(1, 124);
    std::vector<int> civ(1, 124);
    std::vector<int> viv(1, 8);

    FullyDistVec<int, ElementType> ri(riv, G.getcommgrid());
    FullyDistVec<int, ElementType> ci(civ, G.getcommgrid());
    FullyDistVec<int, ElementType> vi(viv, G.getcommgrid());

    PSpMat<ElementType>::MPI_DCCols r_10(nrow, ncol, ri, ci, vi);

    // ==> step 1
    PSpMat<ElementType>::MPI_DCCols m_10(MPI_COMM_WORLD);
    multPrune<RDFINTINT>(G, r_10, m_10, false, true);

    double t1_diag = MPI_Wtime();
    auto dm_10 = diagonalize(m_10);
    double t2_diag = MPI_Wtime();
    if (myrank == 0) {
        cout << "    diagonalize 1 takes : " << (t2_diag - t1_diag) << " s" <<endl;
    }
    mmul_scalar(dm_10, 3);

    // ==> step 2
    PSpMat<ElementType>::MPI_DCCols m_21(MPI_COMM_WORLD);
    multPrune<RDFINTINT>(tG, dm_10, m_21, true, true);

    double t3_trans = MPI_Wtime();
    auto tm_21 = transpose(m_21);
    double t4_trans = MPI_Wtime();
    if (myrank == 0) {
        cout << "    transpose 2 takes : " << (t4_trans - t3_trans) << " s" <<endl;
    }

    double t3_diag = MPI_Wtime();
    auto dm_21 = diagonalize(tm_21);
    double t4_diag = MPI_Wtime();
    if (myrank == 0) {
        cout << "    diagonalize 2 takes : " << (t4_diag - t3_diag) << " s" <<endl;
    }

    // ==> step 3
    multPrune<PTINTINT>(dm_21, m_10, m_10, true, false);

    // end count time
    double t2 = MPI_Wtime();

    if(myrank == 0) {
        cout << "total mult time : " << total_mult_time << " s" << endl;
        cout << "query 2 takes : " << t2 - t1 << " s" << endl;
    }

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

        string Mname("/home/cheny0l/work/db245/fuad/data/lubm320/encoded.mm");
//        string Mname("/project/k1285/fuad/data/lubm320/encoded.mm");
        PSpMat<ElementType>::MPI_DCCols G(MPI_COMM_WORLD);

        double t1 = MPI_Wtime();

        G.ParallelReadMM(Mname, true, maximum<ElementType>());
        G.PrintInfo();

        double t2 = MPI_Wtime();

        if (myrank == 0) {
            cout << "read file takes : " << (t2 - t1) << " s" << endl;
        }

        lubm320_L2(G);
    }

    MPI_Finalize();
    return 0;
}