#include <iostream>
#include <functional>
#include <algorithm>
#include <sstream>
#include <vector>
#include <iterator>
#include <fstream>
#include "../include/CombBLAS.h"

using namespace std;
using namespace combblas;

#define IndexType uint32_t
#define ElementType int

class PSpMat {
public:
    typedef SpDCCols<IndexType, ElementType> DCCols;
    typedef SpParMat<IndexType, ElementType, DCCols> MPI_DCCols;
};


typedef RDFRing<ElementType, ElementType> RDFINTINT;
typedef PlusTimesSRing<ElementType, ElementType> PTINTINT;

static double total_mult_time = 0.0;
static double total_reduce_time = 0.0;
static double total_prune_time = 0.0;
static double total_construct_diag_time = 0.0;
static double total_transpose_time = 0.0;
static double total_mmul_scalar_time = 0.0;

// for constructing diag matrix
static FullyDistVec<int, int> *rvec;
static FullyDistVec<int, int> *qvec;

bool isZero(ElementType t) {
    return t == 0;
}

bool isNotZero(ElementType t) {
    return t != 0;
}

ElementType selectSecond(ElementType a, ElementType b) {
    return b;
}

void permute(PSpMat::MPI_DCCols &G, FullyDistVec<IndexType, ElementType> &nonisov) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // permute G
    double t_perm1 = MPI_Wtime();
    FullyDistVec<IndexType, ElementType> * ColSums = new FullyDistVec<IndexType, ElementType>(G.getcommgrid());
    FullyDistVec<IndexType, ElementType> * RowSums = new FullyDistVec<IndexType, ElementType>(G.getcommgrid());
    G.Reduce(*ColSums, Column, plus<ElementType>(), static_cast<ElementType>(0));
    G.Reduce(*RowSums, Row, plus<ElementType>(), static_cast<ElementType>(0));
    ColSums->EWiseApply(*RowSums, plus<ElementType>());

    nonisov = ColSums->FindInds(bind2nd(greater<ElementType>(), 0));

    nonisov.RandPerm();

    G(nonisov, nonisov, true);
    double t_perm2 = MPI_Wtime();

    float impG = G.LoadImbalance();
    if (myrank == 0) {
        cout << "    permutation takes : " << (t_perm2 - t_perm1) << endl;
        cout << "    imbalance of permuted G : " << impG << endl;
    }
}

void mmul_scalar(PSpMat::MPI_DCCols &M, ElementType s) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t1 = MPI_Wtime();
    M.Apply(bind2nd(multiplies<ElementType>(), s));
    double t2 = MPI_Wtime();

    if (myrank == 0) {
        total_mmul_scalar_time += (t2 - t1);
        cout << "    mmul_scalar takes : " << (t2 - t1) << endl;
    }
}

PSpMat::MPI_DCCols diagonalize(const PSpMat::MPI_DCCols &M, bool isColumn=false) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int dim = M.getnrow();

    FullyDistVec< int, ElementType> diag(M.getcommgrid());

    double t1 = MPI_Wtime();
    if (isColumn) {
        M.Reduce(diag, Column, std::logical_or<ElementType>() , 0);
    } else {
        M.Reduce(diag, Row, std::logical_or<ElementType>() , 0);
    }
    double t2 = MPI_Wtime();

    if (myrank == 0) {
        total_reduce_time += (t2 - t1);
        cout << "    reduce takes : " << (t2 - t1) << endl;
    }

    double t3 = MPI_Wtime();
    PSpMat::MPI_DCCols D(dim, dim, *rvec, *qvec, diag);
    double t4 = MPI_Wtime();

    if (myrank == 0) {
        total_construct_diag_time += (t4 - t3);
        cout << "    construct diag takes : " << (t4 - t3) << endl;
    }

    return D;
}

PSpMat::MPI_DCCols transpose(PSpMat::MPI_DCCols &M) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t1 = MPI_Wtime();

    PSpMat::MPI_DCCols N(M);
    N.Transpose();

    double t2 = MPI_Wtime();
    if (myrank == 0) {
        total_transpose_time += (t2 - t1);
        cout << "    transpose takes " << (t2 - t1) << endl;
    }

    return N;
}

template  <typename  SR>
void multPrune(PSpMat::MPI_DCCols &A, PSpMat::MPI_DCCols &B, PSpMat::MPI_DCCols &C, bool clearA = false, bool clearB = false) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    float imA = A.LoadImbalance(), imB = B.LoadImbalance();
    if (myrank == 0) {
        cout << "    imA : " << imA << "    imB : " << imB << endl;
    }

    double t1 = MPI_Wtime();
    C = Mult_AnXBn_DoubleBuff<SR, ElementType, PSpMat::DCCols>(A, B, clearA, clearB);
    double t2 = MPI_Wtime();

    if (myrank == 0) {
        total_mult_time += (t2 - t1);
        cout << "    multiplication takes: " << (t2 - t1) << " s" << endl;
    }

    double t3 = MPI_Wtime();
    C.Prune(isZero);
    double t4 = MPI_Wtime();

    if (myrank == 0) {
        total_prune_time += (t4 - t3);
        cout << "    prune takes: " << (t4 - t3) << " s\n" << endl;
    }

//    printReducedInfo(C);
}

void printReducedInfo(PSpMat::MPI_DCCols &M){
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

    float imM = M.LoadImbalance();
    if (myrank == 0) {
        cout << "    enum takes " << (t2 - t1) << " s" << endl;
        cout << nnz1 << " [ " << nnzrows1 << ", " << nnzcols1 << " ]" << endl;
        cout << "    imbalance : " << imM << "\n" << endl;
    }
}

void lubm10240_L2(PSpMat::MPI_DCCols &G) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    FullyDistVec<IndexType, ElementType> nonisov(G.getcommgrid());
    permute(G, nonisov);

    double t1_trans = MPI_Wtime();
    auto tG = transpose(G);
    double t2_trans = MPI_Wtime();

    float imtpG = G.LoadImbalance();
    if (myrank == 0) {
        cout << "    transpose G takes : " << (t2_trans - t1_trans) << " s" <<endl;
        cout << "    imbalance of tG : " << imtpG << "\n" << endl;
    }

    double t_cons1 = MPI_Wtime();

    int nrow = G.getnrow(), ncol = G.getncol();
    std::vector<int> riv(1, 79);
    std::vector<int> civ(1, 79);
    std::vector<int> viv(1, 6);

    FullyDistVec<int, ElementType> ri(riv, G.getcommgrid());
    FullyDistVec<int, ElementType> ci(civ, G.getcommgrid());
    FullyDistVec<int, ElementType> vi(viv, G.getcommgrid());

    PSpMat::MPI_DCCols r_10(nrow, ncol, ri, ci, vi);
    r_10(nonisov, nonisov, true);

    double t_cons2 = MPI_Wtime();
    if (myrank == 0) {
        cout << "    construct single element matrix takes : " << (t_cons2 - t_cons1) << "\n" << endl;
    }

    // start count time
    double t1 = MPI_Wtime();

    // ==> step 1
    PSpMat::MPI_DCCols m_10(MPI_COMM_WORLD);
    multPrune<RDFINTINT>(G, r_10, m_10, false, true);

    auto dm_10 = diagonalize(m_10);
    mmul_scalar(dm_10, 3);

    // ==> step 2
    PSpMat::MPI_DCCols m_21(MPI_COMM_WORLD);
    multPrune<RDFINTINT>(tG, dm_10, m_21, true, true);

    auto dm_21 = diagonalize(m_21, true);

    // ==> step 3
    multPrune<PTINTINT>(dm_21, m_10, m_10, true, false);

    // end count time
    double t2 = MPI_Wtime();

    if(myrank == 0) {
        cout << "total mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "total transpose time : " << total_transpose_time << " s" << endl;
        cout << "total prune time : " << total_prune_time << " s" << endl;
        cout << "total reduce time : " << total_reduce_time << " s" << endl;
        cout << "total cons_diag time : " << total_construct_diag_time << " s" << endl;
        cout << "total mult time : " << total_mult_time << " s" << endl;
        cout << "query7 totally takes : " << t2 - t1 << " s" << endl;
    }

}

int main(int argc, char *argv[]) {
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 1) {
        if (myrank == 0) {
            cout << "Usage: ./lubm10240_l2" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    {
        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));

        MPI_Barrier(MPI_COMM_WORLD);

//        string Mname("/home/cheny0l/work/db245/fuad/data/lubm10240/encoded.mm");
        string Mname("/scratch/cheny0l/fuad/data/lubm10240/encoded.mm");
        PSpMat::MPI_DCCols G(MPI_COMM_WORLD);

        double t1 = MPI_Wtime();

        G.ParallelReadMM(Mname, true, selectSecond);
        G.PrintInfo();

        double t2 = MPI_Wtime();

        rvec = new FullyDistVec<int, int>(fullWorld);
        rvec->iota(G.getnrow(), 0);
        qvec = new FullyDistVec<int, int>(fullWorld);
        qvec->iota(G.getnrow(), 0);

        if (myrank == 0) {
            cout << "read file takes : " << (t2 - t1) << " s" << endl;
        }

        lubm10240_L2(G);
    }

    MPI_Finalize();
    return 0;
}
