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
    C = Mult_AnXBn_DoubleBuff<SR, ElementType, PSpMat<ElementType>::DCCols>(A, B, clearA, clearB);
//    C = PSpGEMM<SR>(A, B, clearA, clearB);
    double t2 = MPI_Wtime();

    if (myrank == 0) {
        total_mult_time += (t2 -t1);
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

void lubm320_L7(PSpMat<ElementType>::MPI_DCCols &G) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // start count time
    double total_computing_1 = MPI_Wtime();

    int nrow = G.getnrow(), ncol = G.getncol();
    std::vector<int> riv(1, 107);
    std::vector<int> civ(1, 107);
    std::vector<int> viv(1, 8);

    FullyDistVec<int, ElementType> ri(riv, G.getcommgrid());
    FullyDistVec<int, ElementType> ci(civ, G.getcommgrid());
    FullyDistVec<int, ElementType> vi(viv, G.getcommgrid());

    PSpMat<ElementType>::MPI_DCCols r_30(nrow, ncol, ri, ci, vi);

    // ==> step 1
    PSpMat<ElementType>::MPI_DCCols m_30(MPI_COMM_WORLD);
    multPrune<RDFINTINT>(G, r_30, m_30);

    auto tG = transpose(G);
    auto dm_30 = diagonalize(m_30);
    mmul_scalar(dm_30, 2);

    // ==> step 2
    PSpMat<ElementType>::MPI_DCCols m_43(MPI_COMM_WORLD);
    multPrune<RDFINTINT>(tG, dm_30, m_43);

    auto dm_43 = diagonalize(m_43);
    mmul_scalar(dm_43, 8);

    // ==> step 3
    PSpMat<ElementType>::MPI_DCCols m_14(MPI_COMM_WORLD);
    multPrune<RDFINTINT>(tG, dm_43, m_14);

    int nrow1 = m_14.getnrow(), ncol1 = m_14.getncol();
    std::vector<int> riv1(1, 124);
    std::vector<int> civ1(1, 124);
    std::vector<int> viv1(1, 1);

    FullyDistVec<int, ElementType> ri1(riv1, m_14.getcommgrid());
    FullyDistVec<int, ElementType> ci1(civ1, m_14.getcommgrid());
    FullyDistVec<int, ElementType> vi1(viv1, m_14.getcommgrid());

    PSpMat<ElementType>::MPI_DCCols l_14(nrow1, ncol1, ri1, ci1, vi1);

    // ==> step 4
    multPrune<PTINTINT>(l_14, m_14, m_14);

    auto tm_14 = transpose(m_14);
    auto dm_14 = diagonalize(tm_14);
    mmul_scalar(dm_14, 11);

    // ==> step 5
    PSpMat<ElementType>::MPI_DCCols m_54(MPI_COMM_WORLD);
    multPrune<RDFINTINT>(G, dm_14, m_54);

    auto dm_54 = diagonalize(m_54);
    mmul_scalar(dm_54, 8);

    // ==> step 6
    PSpMat<ElementType>::MPI_DCCols m_25(MPI_COMM_WORLD);
    multPrune<RDFINTINT>(tG, dm_54, m_25);

    int nrow2 = m_25.getnrow(), ncol2 = m_25.getncol();
    std::vector<int> riv2(1, 2079);
    std::vector<int> civ2(1, 2079);
    std::vector<int> viv2(1, 1);

    FullyDistVec<int, ElementType> ri2(riv2, m_25.getcommgrid());
    FullyDistVec<int, ElementType> ci2(civ2, m_25.getcommgrid());
    FullyDistVec<int, ElementType> vi2(viv2, m_25.getcommgrid());

    PSpMat<ElementType>::MPI_DCCols l_25(nrow2, ncol2, ri2, ci2, vi2);

    // ==> step 7
    multPrune<PTINTINT>(l_25, m_25, m_25);

    auto tm_25 = transpose(m_25);
    auto dm_25 = diagonalize(tm_25);
    mmul_scalar(dm_25, 7);

    // ==> step 8
    PSpMat<ElementType>::MPI_DCCols m_65(MPI_COMM_WORLD);
    multPrune<RDFINTINT>(G, dm_25, m_65);

    auto tm_43 = transpose(m_43);
    auto dm_43_1 = diagonalize(tm_43);

    // ==> step 9
    multPrune<PTINTINT>(dm_43_1, m_65, m_65);

    auto dm_65 = diagonalize(m_65);

    // ==> step 10
    multPrune<PTINTINT>(m_43, dm_65, m_43);

    auto tm_65 = transpose(m_65);
    auto dm_65_1 = diagonalize(tm_65);

    // ==> step 11
    multPrune<PTINTINT>(dm_65_1, m_54, m_54);

    auto tm_54 = transpose(m_54);
    auto dm_54_1 = diagonalize(tm_54);

    // ==> step 12
    multPrune<PTINTINT>(dm_54_1, m_43, m_43);

    auto tm_43_1 = transpose(m_43);
    auto dm_43_2 = diagonalize(tm_43_1);

    // ==> step 13
    multPrune<PTINTINT>(dm_43_2, m_30, m_30);

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

        lubm320_L7(G);
    }

    MPI_Finalize();
    return 0;
}