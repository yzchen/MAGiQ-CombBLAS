//
// Created by cheny0l on 10/07/18.
//

#ifndef COMBINATORIAL_BLAS_HEADER100K_H
#define COMBINATORIAL_BLAS_HEADER100K_H
#include <iostream>
#include <functional>
#include <algorithm>
#include <sstream>
#include <vector>
#include <iterator>
#include <fstream>
#include "CombBLAS.h"

using namespace std;
using namespace combblas;

#define IndexType long
//#define ElementType int
#define ElementType long

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
static double total_mmul_scalar_time = 0.0;
static double total_dim_apply_time = 0.0;

// for constructing diag matrix
static FullyDistVec<IndexType, ElementType> *nonisov;

bool isZero(ElementType t) {
    return t == 0;
}

bool isNotZero(ElementType t) {
    return t != 0;
}

ElementType rdf_multiply(ElementType a, ElementType b) {
    if (a != 0 && b != 0 && a == b) {
        return static_cast<ElementType>(1);
    } else {
        return static_cast<ElementType>(0);
    }
}

ElementType selectSecond(ElementType a, ElementType b) {
    return b;
}

void printReducedInfo(PSpMat::MPI_DCCols &M) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t1 = MPI_Wtime();

    int nnz1 = M.getnnz();

    FullyDistVec<IndexType, ElementType> rowsums1(M.getcommgrid());
    M.Reduce(rowsums1, Row, std::plus<ElementType>(), static_cast<ElementType>(0));
    FullyDistVec<IndexType, ElementType> colsums1(M.getcommgrid());
    M.Reduce(colsums1, Column, std::plus<ElementType>(),static_cast<ElementType>(0));
    long nnzrows1 = rowsums1.Count(isNotZero);
    long nnzcols1 = colsums1.Count(isNotZero);

    double t2 = MPI_Wtime();

    float imM = M.LoadImbalance();
    if (myrank == 0) {
        cout << nnz1 << " [ " << nnzrows1 << ", " << nnzcols1 << " ]" << endl;
        cout << "\tenum takes " << (t2 - t1) << " s" << endl;
        cout << "\timbalance : " << imM << endl;
        cout << "---------------------------------------------------------------" << endl;
    }
}

void permute(PSpMat::MPI_DCCols &G) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // permute G
    double t_perm1 = MPI_Wtime();
    FullyDistVec<IndexType, ElementType> *ColSums = new FullyDistVec<IndexType, ElementType>(G.getcommgrid());
    FullyDistVec<IndexType, ElementType> *RowSums = new FullyDistVec<IndexType, ElementType>(G.getcommgrid());
    G.Reduce(*ColSums, Column, plus<ElementType>(), static_cast<ElementType>(0));
    G.Reduce(*RowSums, Row, plus<ElementType>(), static_cast<ElementType>(0));
    ColSums->EWiseApply(*RowSums, plus<ElementType>());

    nonisov = new FullyDistVec<IndexType, ElementType>(G.getcommgrid());
    *nonisov = ColSums->FindInds(bind2nd(greater<ElementType>(), static_cast<ElementType>(0)));

    nonisov->RandPerm();

    G(*nonisov, *nonisov, true);
    double t_perm2 = MPI_Wtime();

    float impG = G.LoadImbalance();
    if (myrank == 0) {
        cout << "\tpermutation takes : " << (t_perm2 - t_perm1) << " s" << endl;
        cout << "\timbalance of permuted G : " << impG << endl;
    }
}

PSpMat::MPI_DCCols transpose(const PSpMat::MPI_DCCols &M) {
    PSpMat::MPI_DCCols N(M);
    N.Transpose();
    return N;
}

void
diagonalizeV(const PSpMat::MPI_DCCols &M, FullyDistVec<IndexType, ElementType> &diag, Dim dim = Row, int scalar = 1) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t1 = MPI_Wtime();
    M.Reduce(diag, dim, std::logical_or<ElementType>(), static_cast<ElementType>(0));
    double t2 = MPI_Wtime();

    double t3 = MPI_Wtime();
    if (scalar != 1) {
        diag.Apply(bind2nd(multiplies<ElementType>(), scalar));
    }
    double t4 = MPI_Wtime();

    if (myrank == 0) {
        total_reduce_time += (t2 - t1);
        total_mmul_scalar_time += (t4 - t3);
        cout << "\tdiag-reduce takes : " << (t2 - t1) << " s" << endl;
        cout << "\tmmul-scalar takes : " << (t4 - t3) << " s" << endl;
    }
}

template<typename SR>
void multPrune(PSpMat::MPI_DCCols &A, PSpMat::MPI_DCCols &B, PSpMat::MPI_DCCols &C, bool clearA = false,
               bool clearB = false) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t1 = MPI_Wtime();
    float imA = A.LoadImbalance(), imB = B.LoadImbalance();
    if (myrank == 0) {
        cout << "\timA : " << imA << "    imB : " << imB << endl;
    }

    C = Mult_AnXBn_DoubleBuff<SR, ElementType, PSpMat::DCCols>(A, B, clearA, clearB);
    double t2 = MPI_Wtime();

    if (myrank == 0) {
        total_mult_time += (t2 - t1);
        cout << "\tmultiplication takes: " << (t2 - t1) << " s" << endl;
    }

    double t3 = MPI_Wtime();
    C.Prune(isZero);
    double t4 = MPI_Wtime();

    if (myrank == 0) {
        total_prune_time += (t4 - t3);
        cout << "\tprune takes: " << (t4 - t3) << " s" << endl;
    }

    // printReducedInfo(C);
}

void multDimApplyPrune(PSpMat::MPI_DCCols &A, FullyDistVec<IndexType, ElementType> &v, Dim dim, bool isRDF) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t1 = MPI_Wtime();
    if (isRDF) {
        A.DimApply(dim, v, rdf_multiply);
    } else {
        A.DimApply(dim, v, std::multiplies<ElementType>());
    }
    double t2 = MPI_Wtime();

    if (myrank == 0) {
        total_dim_apply_time += (t2 - t1);
        cout << "\tdim-apply takes: " << (t2 - t1) << " s" << endl;
    }

    double t3 = MPI_Wtime();
    A.Prune(isZero);
    double t4 = MPI_Wtime();

    if (myrank == 0) {
        total_prune_time += (t4 - t3);
        cout << "\tprune takes: " << (t4 - t3) << " s" << endl;
    }

//    printReducedInfo(A);
}

#endif //COMBINATORIAL_BLAS_HEADER100K_H