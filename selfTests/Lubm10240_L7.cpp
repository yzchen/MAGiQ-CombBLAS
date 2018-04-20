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
    PSpMat<ElementType>::MPI_DCCols N(M);
    N.Transpose();
    return N;
}

bool isZero(ElementType t) {
    return t == 0;
}

bool isNotZero(ElementType t) {
    return t != 0;
}

ElementType selectSecond(ElementType a, ElementType b) {
    return b;
}

static double total_enum_time = 0.0;
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
        total_enum_time += (t2 -t1);
        cout << "    enum takes " << (t2 - t1) << " s" << endl;
    }

    if (myrank == 0) {
        cout << nnz1 << " [ " << nnzrows1 << ", " << nnzcols1 << " ]" << endl;
    }
}

template  <typename  SR>
void multPrune(PSpMat<ElementType>::MPI_DCCols &A, PSpMat<ElementType>::MPI_DCCols &B, PSpMat<ElementType>::MPI_DCCols &C, bool clearA = false, bool clearB = false) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t1 = MPI_Wtime();
//    C = Mult_AnXBn_DoubleBuff<SR, ElementType, PSpMat<ElementType>::DCCols>(A, B, clearA, clearB);
    C = PSpGEMM<SR>(A, B, clearA, clearB);
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

void lubm10240_L7(PSpMat<ElementType>::MPI_DCCols &G) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // start count time
    double total_computing_1 = MPI_Wtime();

    int nrow = G.getnrow(), ncol = G.getncol();
    std::vector<int> riv(1, 1345);
    std::vector<int> civ(1, 1345);
    std::vector<int> viv(1, 6);

    FullyDistVec<int, ElementType> ri(riv, G.getcommgrid());
    FullyDistVec<int, ElementType> ci(civ, G.getcommgrid());
    FullyDistVec<int, ElementType> vi(viv, G.getcommgrid());

    PSpMat<ElementType>::MPI_DCCols r_50(nrow, ncol, ri, ci, vi);

    // ==> step 1
    PSpMat<ElementType>::MPI_DCCols m_50(MPI_COMM_WORLD);
    multPrune<RDFINTINT>(G, r_50, m_50);

    auto dm_50 = diagonalize(m_50);
    mmul_scalar(dm_50, 13);

    // ==> step 2
    PSpMat<ElementType>::MPI_DCCols m_35(MPI_COMM_WORLD);
    multPrune<RDFINTINT>(G, dm_50, m_35);

    auto tG = transpose(G);
    auto dm_35 = diagonalize(m_35);
    mmul_scalar(dm_35, 6);

    // ==> step 3
    PSpMat<ElementType>::MPI_DCCols m_13(MPI_COMM_WORLD);
    multPrune<RDFINTINT>(tG, dm_35, m_13);

    int nrow1 = m_13.getnrow(), ncol1 = m_13.getncol();
    std::vector<int> riv1(1, 43);
    std::vector<int> civ1(1, 43);
    std::vector<int> viv1(1, 1);

    FullyDistVec<int, ElementType> ri1(riv1, m_13.getcommgrid());
    FullyDistVec<int, ElementType> ci1(civ1, m_13.getcommgrid());
    FullyDistVec<int, ElementType> vi1(viv1, m_13.getcommgrid());

    PSpMat<ElementType>::MPI_DCCols l_13(nrow1, ncol1, ri1, ci1, vi1);

    // ==> step 4
    multPrune<PTINTINT>(l_13, m_13, m_13);

    auto tm_13 = transpose(m_13);
    auto dm_13 = diagonalize(tm_13);
    mmul_scalar(dm_13, 8);

    // ==> step 5
    PSpMat<ElementType>::MPI_DCCols m_43(MPI_COMM_WORLD);
    multPrune<RDFINTINT>(tG, dm_13, m_43);

    auto dm_43 = diagonalize(m_43);
    mmul_scalar(dm_43, 6);

    // ==> step 6
    PSpMat<ElementType>::MPI_DCCols m_24(MPI_COMM_WORLD);
    multPrune<RDFINTINT>(tG, dm_43, m_24);

    int nrow2 = m_24.getnrow(), ncol2 = m_24.getncol();
    std::vector<int> riv2(1, 79);
    std::vector<int> civ2(1, 79);
    std::vector<int> viv2(1, 1);

    FullyDistVec<int, ElementType> ri2(riv2, m_24.getcommgrid());
    FullyDistVec<int, ElementType> ci2(civ2, m_24.getcommgrid());
    FullyDistVec<int, ElementType> vi2(viv2, m_24.getcommgrid());

    PSpMat<ElementType>::MPI_DCCols l_24(nrow2, ncol2, ri2, ci2, vi2);

    // ==> step 7
    multPrune<PTINTINT>(l_24, m_24, m_24);

    auto tm_24 = transpose(m_24);
    auto dm_24 = diagonalize(tm_24);
    mmul_scalar(dm_24, 4);

    // ==> step 8
    PSpMat<ElementType>::MPI_DCCols m_64(MPI_COMM_WORLD);
    multPrune<RDFINTINT>(G, dm_24, m_64);

    auto tm_35 = transpose(m_35);
    auto dm_35_1 = diagonalize(tm_35);

    // ==> step 9
    multPrune<PTINTINT>(dm_35_1, m_64, m_64);

    auto dm_64 = diagonalize(m_64);

    // ==> step 10
    multPrune<PTINTINT>(m_35, dm_64, m_35);

    auto tm_64 = transpose(m_64);
    auto dm_64_1 = diagonalize(tm_64);

    // ==> step 11
    multPrune<PTINTINT>(dm_64_1, m_43, m_43);

    auto tm_43 = transpose(m_43);
    auto dm_43_1 = diagonalize(tm_43);

    // ==> step 12
    multPrune<PTINTINT>(dm_43_1, m_35, m_35);

    auto tm_35_1 = transpose(m_35);
    auto dm_35_2 = diagonalize(tm_35_1);

    // ==> step 13
    multPrune<PTINTINT>(dm_35_2, m_50, m_50);

    // end count time
    double total_computing_2 = MPI_Wtime();

    if(myrank == 0) {
        cout << "query 7 totally takes : " << total_computing_2 - total_computing_1 << " s" << endl;
        cout << "total mult time : " << total_mult_time << " s" << endl;
        cout << "total enum time : " << total_enum_time << " s" << endl;
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

        string Mname("/home/cheny0l/work/db245/fuad/data/lubm10240/encoded.mm");
//        string Mname("/project/k1285/fuad/data/lubm10240/encoded.mm");
        PSpMat<ElementType>::MPI_DCCols G(MPI_COMM_WORLD);

        double t1 = MPI_Wtime();

        G.ParallelReadMM(Mname, true, selectSecond);
        G.PrintInfo();

        double t2 = MPI_Wtime();

        if (myrank == 0) {
            cout << "read file takes : " << (t2 - t1) << " s" << endl;
        }

        lubm10240_L7(G);
    }

    MPI_Finalize();
    return 0;
}