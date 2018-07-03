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
static double total_mmul_scalar_time = 0.0;
static double total_dim_apply_time = 0.0;

// for constructing diag matrix
static FullyDistVec<IndexType, ElementType> *rvec;
static FullyDistVec<IndexType, ElementType> *qvec;

bool isZero(ElementType t) {
    return t == 0;
}

bool isNotZero(ElementType t) {
    return t != 0;
}

ElementType rdf_multiply(ElementType a, ElementType b) {
    if (a != 0 &&  b != 0 && a == b) {
        return static_cast<ElementType>(1);
    } else {
        return static_cast<ElementType>(0);
    }
}

ElementType selectSecond(ElementType a, ElementType b) {
    return b;
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
        cout << nnz1 << " [ " << nnzrows1 << ", " << nnzcols1 << " ]" << endl;
        cout << "\tenum takes " << (t2 - t1) << " s" << endl;
        cout << "\timbalance : " << imM << endl;
        cout << "---------------------------------------------------------------" << endl;
    }
}

void permute(PSpMat::MPI_DCCols &G, FullyDistVec<IndexType, IndexType> &nonisov) {
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
        cout << "\tpermutation takes : " << (t_perm2 - t_perm1) << " s" << endl;
        cout << "\timbalance of permuted G : " << impG << endl;
    }
}

PSpMat::MPI_DCCols transpose(const PSpMat::MPI_DCCols &M) {
    PSpMat::MPI_DCCols N(M);
    N.Transpose();
    return N;
}

void diagonalizeV(const PSpMat::MPI_DCCols &M, FullyDistVec<IndexType, ElementType> &diag, Dim dim=Row, int scalar=1) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t1 = MPI_Wtime();
    M.Reduce(diag, dim, std::logical_or<ElementType>() , 0);
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

void diagonalizeM(const PSpMat::MPI_DCCols &M, PSpMat::MPI_DCCols &D, Dim dim=Row, int scalar=1) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t1 = MPI_Wtime();
    int dimV = M.getnrow();
    FullyDistVec< int, ElementType> diag(M.getcommgrid());
    M.Reduce(diag, dim, std::logical_or<ElementType>() , 0);
    double t2 = MPI_Wtime();

    double t3 = MPI_Wtime();
    D = PSpMat::MPI_DCCols(dimV, dimV, *rvec, *qvec, diag);
    double t4 = MPI_Wtime();

    double t5 = MPI_Wtime();
    if (scalar != 1) {
        D.Apply(bind2nd(multiplies<ElementType>(), scalar));
    }
    double t6 = MPI_Wtime();

    if (myrank == 0) {
        total_reduce_time += (t2 - t1);
        total_construct_diag_time += (t4 - t3);
        total_mmul_scalar_time += (t6 - t5);
        cout << "\treduce takes : " << (t2 - t1) << " s" << endl;
        cout << "\tconstruct diag takes : " << (t4 - t3) << " s" << endl;
        cout << "\tmmul-scalar takes : " << (t6 - t5) << " s" << endl;
    }
}

template  <typename  SR>
void multPrune(PSpMat::MPI_DCCols &A, PSpMat::MPI_DCCols &B, PSpMat::MPI_DCCols &C, bool clearA = false, bool clearB = false) {
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

void lubm10240_l1(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG, FullyDistVec<IndexType, IndexType> &nonisov) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    total_reduce_time = 0.0;
    total_prune_time = 0.0;
    total_construct_diag_time = 0.0;
    total_mmul_scalar_time = 0.0;
    total_dim_apply_time = 0.0;

    auto commWorld = G.getcommgrid();

    FullyDistVec<IndexType, ElementType> dm_50(commWorld), dm_35(commWorld), dm_13(commWorld),
            dm_43(commWorld), dm_24(commWorld), dm_35_1(commWorld), dm_64(commWorld), dm_64_1(commWorld),
            dm_43_1(commWorld), dm_35_2(commWorld);

    auto m_50(G), m_35(G), m_13(tG), m_43(tG), m_24(tG), m_64(G);

    IndexType ind1 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(22638)))[0];
    IndexType ind2 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(24)))[0];
    IndexType ind3 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(8622222)))[0];

    FullyDistVec<IndexType, ElementType> r_50(commWorld, G.getnrow(), 0), l_13(commWorld, G.getnrow(), 0), l_24(commWorld, G.getnrow(), 0);
    r_50.SetElement(ind1, 6);
    l_13.SetElement(ind2, 1);
    l_24.SetElement(ind3, 1);

    // start count time
    double total_computing_1 = MPI_Wtime();

    // ==> step 1
    if (myrank == 0) {
        cout << "\n###############################################################" << endl;
        cout << "Query 1" << endl;
        cout << "###############################################################" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "step 1 : m_(5,0) = G x {1@(1345,1345)}*6" << endl;
    }
    double t1_start = MPI_Wtime();
    multDimApplyPrune(m_50, r_50, Column, true);
    double t1_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 1 (Total) : " << (t1_end - t1_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 2
    if (myrank == 0) {
        cout << "step 2 : m_(3, 5) = G * m_(5, 0).D()*13" << endl;
    }
    double t2_start = MPI_Wtime();
    diagonalizeV(m_50, dm_50, Row, 10);
    multDimApplyPrune(m_35, dm_50, Column, true);
    double t2_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 2 (Total) : " << (t2_end - t2_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 3
    if (myrank == 0) {
        cout << "step 3 : m_(1,3) = G.T() x m_(3,5).D()*6" << endl;
    }
    double t3_start = MPI_Wtime();
    diagonalizeV(m_35, dm_35, Row, 6);
    multDimApplyPrune(m_13, dm_35, Column, true);
    double t3_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 3 (Total) : " << (t3_end - t3_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 4
    if (myrank == 0) {
        cout << "step 4 : m_(1,3) = {1@(43,43)} x m_(1,3)" << endl;
    }
    double t4_start = MPI_Wtime();
    multDimApplyPrune(m_13, l_13, Row, false);
    double t4_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 4 (Total) : " << (t4_end - t4_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 5
    if (myrank == 0) {
        cout << "step 5 : m_(4,3) = G.T() x m_(1,3).T().D()*8" << endl;
    }
    double t5_start = MPI_Wtime();
    diagonalizeV(m_13, dm_13, Column, 2);
    multDimApplyPrune(m_43, dm_13, Column, true);
    double t5_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 5 (Total) : " << (t5_end - t5_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 6
    if (myrank == 0) {
        cout << "step 6 : m_(2,4) = G.T() x m_(4,3).D()*6" << endl;
    }
    double t6_start = MPI_Wtime();
    diagonalizeV(m_43, dm_43, Row, 6);
    multDimApplyPrune(m_24, dm_43, Column, true);
    double t6_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 6 (Total) : " << (t6_end - t6_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 7
    if (myrank == 0) {
        cout << "step 7 : m_(2,4) = {1@(79,79)} x m_(2,4)" << endl;
    }
    double t7_start = MPI_Wtime();
    multDimApplyPrune(m_24, l_24, Row, false);
    double t7_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 7 (Total) : " << (t7_end - t7_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 8
    if (myrank == 0) {
        cout << "step 8 : m_(6,4) = G x m_(2,4).T().D()*4" << endl;
    }
    double t8_start = MPI_Wtime();
    diagonalizeV(m_24, dm_24, Column, 11);
    multDimApplyPrune(m_64, dm_24, Column, true);
    double t8_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 8 (Total) : " << (t8_end - t8_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 9
    if (myrank == 0) {
        cout << "step 9 : m_(6,4) = m_(3,5).T().D() x m_(6,4)" << endl;
    }
    double t9_start = MPI_Wtime();
    diagonalizeV(m_35, dm_35_1, Column);
    multDimApplyPrune(m_64, dm_35_1, Row, false);
    double t9_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 9 (Total) : " << (t9_end - t9_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 10
    if (myrank == 0) {
        cout << "step 10 : m_(3,5) = m_(3,5) x m_(6,4).D()" << endl;
    }
    double t10_start = MPI_Wtime();
    diagonalizeV(m_64, dm_64);
    multDimApplyPrune(m_35, dm_64, Column, false);
    double t10_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 10 (Total) : " << (t10_end - t10_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 11
    if (myrank == 0) {
        cout << "step 11 : m_(4,3) = m_(6,4).T().D() x m_(4,3)" << endl;
    }
    double t11_start = MPI_Wtime();
    diagonalizeV(m_64, dm_64_1, Column);
    multDimApplyPrune(m_43, dm_64_1, Row, false);
    double t11_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 11 (Total) : " << (t11_end - t11_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 12
    if (myrank == 0) {
        cout << "step 12 : m_(3,5) = m_(4,3).T().D() x m_(3,5)" << endl;
    }
    double t12_start = MPI_Wtime();
    diagonalizeV(m_43, dm_43_1, Column);
    multDimApplyPrune(m_35, dm_43_1, Row, false);
    double t12_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 12 (Total) : " << (t12_end - t12_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 13
    if (myrank == 0) {
        cout << "step 13 : m_(5,0) = m_(3,5).T().D() x m_(5,0)" << endl;
    }
    double t13_start = MPI_Wtime();
    diagonalizeV(m_35, dm_35_2, Column);
    multDimApplyPrune(m_50, dm_35_2, Row, false);
    double t13_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 13 (Total) : " << (t13_end - t13_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // end count time
    double total_computing_2 = MPI_Wtime();

    printReducedInfo(m_50);

    if (myrank == 0) {
        cout << "query1 mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "query1 prune time : " << total_prune_time << " s" << endl;
        cout << "query1 diag_reduce time : " << total_reduce_time << " s" << endl;
        cout << "query1 dim_apply time : " << total_dim_apply_time << " s" << endl;
        cout << "query1 time (Total) : " << total_computing_2 - total_computing_1 << " s" << endl;
    }
}

void lubm10240_l2(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG, FullyDistVec<IndexType, IndexType> &nonisov) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    total_reduce_time = 0.0;
    total_prune_time = 0.0;
    total_construct_diag_time = 0.0;
    total_mmul_scalar_time = 0.0;
    total_dim_apply_time = 0.0;

    auto commWorld = G.getcommgrid();

    FullyDistVec<IndexType, ElementType> dm_10(commWorld), dm_21(commWorld);

    auto m_10(G), m_21(tG);

    IndexType ind1 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(79)))[0];

    FullyDistVec<IndexType, ElementType> r_10(commWorld, G.getnrow(), 0);
    r_10.SetElement(ind1, 6);

    // start count time
    double total_computing_1 = MPI_Wtime();

    // ==> step 1
    if (myrank == 0) {
        cout << "\n###############################################################" << endl;
        cout << "Query 2" << endl;
        cout << "###############################################################" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "step 1 : m_(1,0) = G x {1@(79,79)}*6" << endl;
    }
    double t1_start = MPI_Wtime();
    multDimApplyPrune(m_10, r_10, Column, true);
    double t1_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 1 (Total) : " << (t1_end - t1_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 2
    if (myrank == 0) {
        cout << "step 2 : m_(2,1) = G.T() * m_(1,0).D()*3" << endl;
    }
    double t2_start = MPI_Wtime();
    diagonalizeV(m_10, dm_10, Row, 3);
    multDimApplyPrune(m_21, dm_10, Column, true);
    double t2_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 2 (Total) : " << (t2_end - t2_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 3
    if (myrank == 0) {
        cout << "step 3 : m_(1,0) = m_(2,1).T().D() x m_(1,0)" << endl;
    }
    double t3_start = MPI_Wtime();
    diagonalizeV(m_21, dm_21, Column);
    multDimApplyPrune(m_10, dm_21, Row, false);
    double t3_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 3 (Total) : " << (t3_end - t3_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // end count time
    double total_computing_2 = MPI_Wtime();

    printReducedInfo(m_10);

    if (myrank == 0) {
        cout << "query2 mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "query2 prune time : " << total_prune_time << " s" << endl;
        cout << "query2 diag_reduce time : " << total_reduce_time << " s" << endl;
        cout << "query2 dim_apply time : " << total_dim_apply_time << " s" << endl;
        cout << "query2 time (Total) : " << total_computing_2 - total_computing_1 << " s" << endl;
    }
}

void lubm10240_l3(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG, FullyDistVec<IndexType, IndexType> &nonisov) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    total_reduce_time = 0.0;
    total_prune_time = 0.0;
    total_construct_diag_time = 0.0;
    total_mmul_scalar_time = 0.0;
    total_dim_apply_time = 0.0;

    auto commWorld = G.getcommgrid();

    FullyDistVec<IndexType, ElementType> dm_50(commWorld), dm_35(commWorld), dm_13(commWorld),
            dm_43(commWorld), dm_24(commWorld), dm_35_1(commWorld), dm_64(commWorld), dm_64_1(commWorld),
            dm_43_1(commWorld), dm_35_2(commWorld);

    auto m_50(G), m_35(G), m_13(tG), m_43(tG), m_24(tG), m_64(G);

    IndexType ind1 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(22638)))[0];
    IndexType ind2 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(43)))[0];
    IndexType ind3 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(8622222)))[0];

    FullyDistVec<IndexType, ElementType> r_50(commWorld, G.getnrow(), 0), l_13(commWorld, G.getnrow(), 0), l_24(commWorld, G.getnrow(), 0);
    r_50.SetElement(ind1, 6);
    l_13.SetElement(ind2, 1);
    l_24.SetElement(ind3, 1);

    // start count time
    double total_computing_1 = MPI_Wtime();

    // ==> step 1
    if (myrank == 0) {
        cout << "\n###############################################################" << endl;
        cout << "Query 3" << endl;
        cout << "###############################################################" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "step 1 : m_(5,0) = G x {1@(22638,22638)}*6" << endl;
    }
    double t1_start = MPI_Wtime();
    multDimApplyPrune(m_50, r_50, Column, true);
    double t1_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 1 (Total) : " << (t1_end - t1_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 2
    if (myrank == 0) {
        cout << "step 2 : m_(3, 5) = G * m_(5, 0).D()*10" << endl;
    }
    double t2_start = MPI_Wtime();
    diagonalizeV(m_50, dm_50, Row, 10);
    multDimApplyPrune(m_35, dm_50, Column, true);
    double t2_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 2 (Total) : " << (t2_end - t2_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 3
    if (myrank == 0) {
        cout << "step 3 : m_(1,3) = G.T() x m_(3,5).D()*6" << endl;
    }
    double t3_start = MPI_Wtime();
    diagonalizeV(m_35, dm_35, Row, 6);
    multDimApplyPrune(m_13, dm_35, Column, true);
    double t3_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 3 (Total) : " << (t3_end - t3_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 4
    if (myrank == 0) {
        cout << "step 4 : m_(1,3) = {1@(43,43)} x m_(1,3)" << endl;
    }
    double t4_start = MPI_Wtime();
    multDimApplyPrune(m_13, l_13, Row, false);
    double t4_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 4 (Total) : " << (t4_end - t4_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 5
    if (myrank == 0) {
        cout << "step 5 : m_(4,3) = G.T() x m_(1,3).T().D()*2" << endl;
    }
    double t5_start = MPI_Wtime();
    diagonalizeV(m_13, dm_13, Column, 2);
    multDimApplyPrune(m_43, dm_13, Column, true);
    double t5_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 5 (Total) : " << (t5_end - t5_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

//    // ==> step 6
//    if (myrank == 0) {
//        cout << "step 6 : m_(2,4) = G.T() x m_(4,3).D()*6" << endl;
//    }
//    double t6_start = MPI_Wtime();
//    diagonalizeV(m_43, dm_43, Row, 6);
//    multDimApplyPrune(m_24, dm_43, Column, true);
//    double t6_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 6 (Total) : " << (t6_end - t6_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }
//
//    // ==> step 7
//    if (myrank == 0) {
//        cout << "step 7 : m_(2,4) = {1@(862222,862222)} x m_(2,4)" << endl;
//    }
//    double t7_start = MPI_Wtime();
//    multDimApplyPrune(m_24, l_24, Row, false);
//    double t7_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 7 (Total) : " << (t7_end - t7_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }
//
//    // ==> step 8
//    if (myrank == 0) {
//        cout << "step 8 : m_(6,4) = G x m_(2,4).T().D()*11" << endl;
//    }
//    double t8_start = MPI_Wtime();
//    diagonalizeV(m_24, dm_24, Column, 11);
//    multDimApplyPrune(m_64, dm_24, Column, true);
//    double t8_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 8 (Total) : " << (t8_end - t8_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }
//
//    // ==> step 9
//    if (myrank == 0) {
//        cout << "step 9 : m_(6,4) = m_(3,5).T().D() x m_(6,4)" << endl;
//    }
//    double t9_start = MPI_Wtime();
//    diagonalizeV(m_35, dm_35_1, Column);
//    multDimApplyPrune(m_64, dm_35_1, Row, false);
//    double t9_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 9 (Total) : " << (t9_end - t9_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }
//
//    // ==> step 10
//    if (myrank == 0) {
//        cout << "step 10 : m_(3,5) = m_(3,5) x m_(6,4).D()" << endl;
//    }
//    double t10_start = MPI_Wtime();
//    diagonalizeV(m_64, dm_64);
//    multDimApplyPrune(m_35, dm_64, Column, false);
//    double t10_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 10 (Total) : " << (t10_end - t10_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }
//
//    // ==> step 11
//    if (myrank == 0) {
//        cout << "step 11 : m_(4,3) = m_(6,4).T().D() x m_(4,3)" << endl;
//    }
//    double t11_start = MPI_Wtime();
//    diagonalizeV(m_64, dm_64_1, Column);
//    multDimApplyPrune(m_43, dm_64_1, Row, false);
//    double t11_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 11 (Total) : " << (t11_end - t11_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }
//
//    // ==> step 12
//    if (myrank == 0) {
//        cout << "step 12 : m_(3,5) = m_(4,3).T().D() x m_(3,5)" << endl;
//    }
//    double t12_start = MPI_Wtime();
//    diagonalizeV(m_43, dm_43_1, Column);
//    multDimApplyPrune(m_35, dm_43_1, Row, false);
//    double t12_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 12 (Total) : " << (t12_end - t12_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }
//
//    // ==> step 13
//    if (myrank == 0) {
//        cout << "step 13 : m_(5,0) = m_(3,5).T().D() x m_(5,0)" << endl;
//    }
//    double t13_start = MPI_Wtime();
//    diagonalizeV(m_35, dm_35_2, Column);
//    multDimApplyPrune(m_50, dm_35_2, Row, false);
//    double t13_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 13 (Total) : " << (t13_end - t13_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }

    // end count time
    double total_computing_2 = MPI_Wtime();

    printReducedInfo(m_43);

    if (myrank == 0) {
        cout << "query3 mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "query3 prune time : " << total_prune_time << " s" << endl;
        cout << "query3 diag_reduce time : " << total_reduce_time << " s" << endl;
        cout << "query3 dim_apply time : " << total_dim_apply_time << " s" << endl;
        cout << "query3 time (Total) : " << total_computing_2 - total_computing_1 << " s" << endl;
    }
}

void lubm10240_l4(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG, FullyDistVec<IndexType, IndexType> &nonisov) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    total_reduce_time = 0.0;
    total_prune_time = 0.0;
    total_construct_diag_time = 0.0;
    total_mmul_scalar_time = 0.0;
    total_dim_apply_time = 0.0;

    auto commWorld = G.getcommgrid();

    FullyDistVec<IndexType, ElementType> dm_20(commWorld), dm_12(commWorld), dm_32(commWorld), dm_42(commWorld), dm_52(commWorld);

    auto m_20(G), m_12(tG), m_32(tG), m_42(tG), m_52(tG);

    IndexType ind1 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(11)))[0];
    IndexType ind2 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(1345)))[0];

    FullyDistVec<IndexType, ElementType> r_20(commWorld, G.getnrow(), 0), l_12(commWorld, G.getnrow(), 0);
    r_20.SetElement(ind1, 5);
    l_12.SetElement(ind2, 1);

    // start count time
    double total_computing_1 = MPI_Wtime();

    // ==> step 1
    if (myrank == 0) {
        cout << "\n###############################################################" << endl;
        cout << "Query 4" << endl;
        cout << "###############################################################" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "step 1 : m_(2,0) = G x {1@(11,11)}*5" << endl;
    }
    double t1_start = MPI_Wtime();
    multDimApplyPrune(m_20, r_20, Column, true);
    double t1_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 1 (Total) : " << (t1_end - t1_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 2
    if (myrank == 0) {
        cout << "step 2 : m_(1,2) = G.T() x m_(2,0).D()*6" << endl;
    }
    double t2_start = MPI_Wtime();
    diagonalizeV(m_20, dm_20, Row, 6);
    multDimApplyPrune(m_12, dm_20, Column, true);
    double t2_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 2 (Total) : " << (t2_end - t2_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 3
    if (myrank == 0) {
        cout << "step 3 : m_(1,0) = m_(2,1).T().D() x m_(1,0)" << endl;
    }
    double t3_start = MPI_Wtime();
    multDimApplyPrune(m_12, l_12, Row, false);
    double t3_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 3 (Total) : " << (t3_end - t3_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 4
    if (myrank == 0) {
        cout << "step 4 : m_(3,2) = G.T() x m_(1,2).T().D()*3" << endl;
    }
    double t4_start = MPI_Wtime();
    diagonalizeV(m_12, dm_12, Column, 3);
    multDimApplyPrune(m_32, dm_12, Column, true);
    double t4_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 4 (Total) : " << (t4_end - t4_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 5
    if (myrank == 0) {
        cout << "step 5 : m_(4,2) = G.T() x m_(3,2).T().D()*12" << endl;
    }
    double t5_start = MPI_Wtime();
    diagonalizeV(m_32, dm_32, Column, 12);
    multDimApplyPrune(m_42, dm_32, Column, true);
    double t5_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 5 (Total) : " << (t5_end - t5_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 6
    if (myrank == 0) {
        cout << "step 6 : m_(5,2) = G.T() x m_(4,2).T().D()*9" << endl;
    }
    double t6_start = MPI_Wtime();
    diagonalizeV(m_42, dm_42, Column, 9);
    multDimApplyPrune(m_52, dm_42, Column, true);
    double t6_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 6 (Total) : " << (t6_end - t6_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 7
    if (myrank == 0) {
        cout << "step 7 : m_(2,0) = m_(5,2).T().D() x m_(2,0)" << endl;
    }
    double t7_start = MPI_Wtime();
    diagonalizeV(m_52, dm_52, Column);
    multDimApplyPrune(m_20, dm_52, Row, true);
    double t7_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 7 (Total) : " << (t7_end - t7_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // end count time
    double total_computing_2 = MPI_Wtime();

    printReducedInfo(m_20);

    if (myrank == 0) {
        cout << "query4 mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "query4 prune time : " << total_prune_time << " s" << endl;
        cout << "query4 diag_reduce time : " << total_reduce_time << " s" << endl;
        cout << "query4 dim_apply time : " << total_dim_apply_time << " s" << endl;
        cout << "query4 time (Total) : " << total_computing_2 - total_computing_1 << " s" << endl;
    }
}

void lubm10240_l5(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG, FullyDistVec<IndexType, IndexType> &nonisov) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    total_reduce_time = 0.0;
    total_prune_time = 0.0;
    total_construct_diag_time = 0.0;
    total_mmul_scalar_time = 0.0;
    total_dim_apply_time = 0.0;

    auto commWorld = G.getcommgrid();

    FullyDistVec<IndexType, ElementType> dm_20(commWorld), dm_12(commWorld);

    auto m_20(G), m_12(tG);

    IndexType ind1 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(11)))[0];
    IndexType ind2 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(357)))[0];

    FullyDistVec<IndexType, ElementType> r_20(commWorld, G.getnrow(), 0), l_12(commWorld, G.getnrow(), 0);
    r_20.SetElement(ind1, 11);
    l_12.SetElement(ind2, 1);

    // start count time
    double total_computing_1 = MPI_Wtime();

    // ==> step 1
    if (myrank == 0) {
        cout << "\n###############################################################" << endl;
        cout << "Query 5" << endl;
        cout << "###############################################################" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "step 1 : m_(2,0) = G x {1@(11,11)}*11" << endl;
    }
    double t1_start = MPI_Wtime();
    multDimApplyPrune(m_20, r_20, Column, true);
    double t1_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 1 (Total) : " << (t1_end - t1_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 2
    if (myrank == 0) {
        cout << "step 2 : G.T() x m_(2,0).D()*6" << endl;
    }
    double t2_start = MPI_Wtime();
    diagonalizeV(m_20, dm_20, Row, 6);
    multDimApplyPrune(m_12, dm_20, Column, true);
    double t2_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 2 (Total) : " << (t2_end - t2_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 3
    if (myrank == 0) {
        cout << "step 3 : m_(1,2) = {1@(357,357)} x m_(1,2)" << endl;
    }
    double t3_start = MPI_Wtime();
    multDimApplyPrune(m_12, l_12, Row, false);
    double t3_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 3 (Total) : " << (t3_end - t3_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 4
    if (myrank == 0) {
        cout << "step 4 : m_(2,0) = m_(1,2).T().D() x m_(2,0)" << endl;
    }
    double t4_start = MPI_Wtime();
    diagonalizeV(m_12, dm_12, Column);
    multDimApplyPrune(m_20, dm_12, Row, true);
    double t4_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 4 (Total) : " << (t4_end - t4_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // end count time
    double total_computing_2 = MPI_Wtime();

    printReducedInfo(m_20);

    if (myrank == 0) {
        cout << "query5 mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "query5 prune time : " << total_prune_time << " s" << endl;
        cout << "query5 diag_reduce time : " << total_reduce_time << " s" << endl;
        cout << "query5 dim_apply time : " << total_dim_apply_time << " s" << endl;
        cout << "query5 time (Total) : " << total_computing_2 - total_computing_1 << " s" << endl;
    }
}

void lubm10240_l6(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG, FullyDistVec<IndexType, IndexType> &nonisov) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    total_reduce_time = 0.0;
    total_prune_time = 0.0;
    total_construct_diag_time = 0.0;
    total_mmul_scalar_time = 0.0;
    total_dim_apply_time = 0.0;

    auto commWorld = G.getcommgrid();

    FullyDistVec<IndexType, ElementType> dm_30(commWorld), dm_43(commWorld), dm_14(commWorld), dm_24(commWorld);

    auto m_30(G), m_43(tG), m_14(tG), m_24(tG);

    IndexType ind1 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(1345)))[0];
    IndexType ind2 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(22638)))[0];
    IndexType ind3 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(40169)))[0];

    FullyDistVec<IndexType, ElementType> r_30(commWorld, G.getnrow(), 0), l_14(commWorld, G.getnrow(), 0), l_24(commWorld, G.getnrow(), 0);
    r_30.SetElement(ind1, 6);
    l_14.SetElement(ind2, 1);
    l_24.SetElement(ind3, 1);

    // start count time
    double total_computing_1 = MPI_Wtime();

    // ==> step 1
    if (myrank == 0) {
        cout << "\n###############################################################" << endl;
        cout << "Query 6" << endl;
        cout << "###############################################################" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "step 1 : m_(3,0) = G x {1@(1345,1345)}*6" << endl;
    }
    double t1_start = MPI_Wtime();
    multDimApplyPrune(m_30, r_30, Column, true);
    double t1_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 1 (Total) : " << (t1_end - t1_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 2
    if (myrank == 0) {
        cout << "step 2 : m_(4,3) = G.T() x m_(3,0).D()*5" << endl;
    }
    double t2_start = MPI_Wtime();
    diagonalizeV(m_30, dm_30, Row, 5);
    multDimApplyPrune(m_43, dm_30, Column, true);
    double t2_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 2 (Total) : " << (t2_end - t2_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 3
    if (myrank == 0) {
        cout << "step 3 : m_(1,4) = G.T() x m_(4,3).D()*6" << endl;
    }
    double t3_start = MPI_Wtime();
    diagonalizeV(m_43, dm_43, Row, 6);
    multDimApplyPrune(m_14, dm_43, Column, true);
    double t3_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 3 (Total) : " << (t3_end - t3_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 4
    if (myrank == 0) {
        cout << "step 4 : m_(1,4) = {1@(22638,22638)} x m_(1,4)" << endl;
    }
    double t4_start = MPI_Wtime();
    multDimApplyPrune(m_14, l_14, Row, false);
    double t4_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 4 (Total) : " << (t4_end - t4_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 5
    if (myrank == 0) {
        cout << "step 5 : m_(2,4) = G.T() x m_(1,4).T().D()*11" << endl;
    }
    double t5_start = MPI_Wtime();
    diagonalizeV(m_14, dm_14, Column, 11);
    multDimApplyPrune(m_24, dm_14, Column, true);
    double t5_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 5 (Total) : " << (t5_end - t5_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 6
    if (myrank == 0) {
        cout << "step 6 : m_(2,4) = {1@(40169,40169)} x m_(2,4)" << endl;
    }
    double t6_start = MPI_Wtime();
    multDimApplyPrune(m_24, l_24, Row, false);
    double t6_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 6 (Total) : " << (t6_end - t6_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 7
    if (myrank == 0) {
        cout << "step 7 : m_(4,3) = m_(2,4).T().D() x m_(4,3)" << endl;
    }
    double t7_start = MPI_Wtime();
    diagonalizeV(m_24, dm_24, Column);
    multDimApplyPrune(m_43, dm_24, Row, true);
    double t7_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 7 (Total) : " << (t7_end - t7_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 8
    if (myrank == 0) {
        cout << "step 8 : m_(3,0) = m_(4,3).T().D() x m_(3,0)" << endl;
    }
    double t8_start = MPI_Wtime();
    diagonalizeV(m_43, dm_43, Column);
    multDimApplyPrune(m_30, dm_43, Row, true);
    double t8_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 8 (Total) : " << (t8_end - t8_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // end count time
    double total_computing_2 = MPI_Wtime();

    printReducedInfo(m_30);

    if (myrank == 0) {
        cout << "query6 mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "query6 prune time : " << total_prune_time << " s" << endl;
        cout << "query6 diag_reduce time : " << total_reduce_time << " s" << endl;
        cout << "query6 dim_apply time : " << total_dim_apply_time << " s" << endl;
        cout << "query6 time (Total) : " << total_computing_2 - total_computing_1 << " s" << endl;
    }
}

// you have two choices
// choice == 0 : use multPrune
// choice == 1 : use multDimApplyPrune
void lubm10240_l7(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG, FullyDistVec<IndexType, IndexType> &nonisov, int choice=1) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    total_reduce_time = 0.0;
    total_prune_time = 0.0;
    total_construct_diag_time = 0.0;
    total_mmul_scalar_time = 0.0;
    total_dim_apply_time = 0.0;

    auto commWorld = G.getcommgrid();

    IndexType ind1 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(1345)))[0];
    IndexType ind2 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(43)))[0];
    IndexType ind3 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(79)))[0];

    // query execution
    if (choice == 0){   // multPrune
        double t_cons1 = MPI_Wtime();

        int nrow = G.getnrow(), ncol = G.getncol();
        std::vector<int> riv(1, ind1);
        std::vector<int> viv(1, 6);
        FullyDistVec<int, ElementType> ri(riv, G.getcommgrid());
        FullyDistVec<int, ElementType> vi(viv, G.getcommgrid());
        PSpMat::MPI_DCCols r_50(nrow, ncol, ri, ri, vi);

        std::vector<int> riv1(1, ind2);
        std::vector<int> viv1(1, 1);
        FullyDistVec<int, ElementType> ri1(riv1, G.getcommgrid());
        FullyDistVec<int, ElementType> vi1(viv1, G.getcommgrid());
        PSpMat::MPI_DCCols l_13(nrow, ncol, ri1, ri1, vi1);

        std::vector<int> riv2(1, ind3);
        std::vector<int> viv2(1, 1);
        FullyDistVec<int, ElementType> ri2(riv2, G.getcommgrid());
        FullyDistVec<int, ElementType> vi2(viv2, G.getcommgrid());
        PSpMat::MPI_DCCols l_24(nrow, ncol, ri2, ri2, vi2);

        double t_cons2 = MPI_Wtime();
        if (myrank == 0) {
            cout << "construct single element matrix takes : " << (t_cons2 - t_cons1) << " s\n" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        PSpMat::MPI_DCCols dm_50(commWorld), dm_35(commWorld), dm_13(commWorld),
                dm_43(commWorld), dm_24(commWorld), dm_35_1(commWorld), dm_64(commWorld), dm_64_1(commWorld),
                dm_43_1(commWorld), dm_35_2(commWorld);

        PSpMat::MPI_DCCols m_50(commWorld), m_35(commWorld), m_13(commWorld),
                m_43(commWorld), m_24(commWorld), m_64(commWorld);

        // start count time
        double total_computing_1 = MPI_Wtime();

        // ==> step 1
        if (myrank == 0) {
            cout << "step 1 : m_(5,0) = G x {1@(1345,1345)}*6" << endl;
        }
        double t1_start = MPI_Wtime();
        multPrune<RDFINTINT>(G, r_50, m_50);
        double t1_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 1 (Total) : " << (t1_end - t1_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // ==> step 2
        if (myrank == 0) {
            cout << "step 2 : m_(3, 5) = G x m_(5, 0).D()*13" << endl;
        }
        double t2_start = MPI_Wtime();
        diagonalizeM(m_50, dm_50, Row, 13);
        multPrune<RDFINTINT>(G, dm_50, m_35);
        double t2_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 2 (Total) : " << (t2_end - t2_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // ==> step 3
        if (myrank == 0) {
            cout << "step 3 : m_(1,3) = G.T() x m_(3,5).D()*6" << endl;
        }
        double t3_start = MPI_Wtime();
        diagonalizeM(m_35, dm_35, Row, 6);
        multPrune<RDFINTINT>(tG, dm_35, m_13);
        double t3_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 3 (Total) : " << (t3_end - t3_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // ==> step 4
        if (myrank == 0) {
            cout << "step 4 : m_(1,3) = {1@(43,43)} x m_(1,3)" << endl;
        }
        double t4_start = MPI_Wtime();
        multPrune<PTINTINT>(l_13, m_13, m_13);
        double t4_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 4 (Total) : " << (t4_end - t4_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // ==> step 5
        if (myrank == 0) {
            cout << "step 5 : m_(4,3) = G.T() x m_(1,3).T().D()*8" << endl;
        }
        double t5_start = MPI_Wtime();
        diagonalizeM(m_13, dm_13, Column, 8);
        multPrune<RDFINTINT>(tG, dm_13, m_43);
        double t5_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 5 (Total) : " << (t5_end - t5_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // ==> step 6
        if (myrank == 0) {
            cout << "step 6 : m_(2,4) = G.T() x m_(4,3).D()*6" << endl;
        }
        double t6_start = MPI_Wtime();
        diagonalizeM(m_43, dm_43, Row, 6);
        multPrune<RDFINTINT>(tG, dm_43, m_24);
        double t6_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 6 (Total) : " << (t6_end - t6_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // ==> step 7
       if (myrank == 0) {
            cout << "step 7 : m_(2,4) = {1@(79,79)} x m_(2,4)" << endl;
        }
        double t7_start = MPI_Wtime();
        multPrune<PTINTINT>(l_24, m_24, m_24);
        double t7_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 7 (Total) : " << (t7_end - t7_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // ==> step 8
        if (myrank == 0) {
            cout << "step 8 : m_(6,4) = G x m_(2,4).T().D()*4" << endl;
        }
        double t8_start = MPI_Wtime();
        diagonalizeM(m_24, dm_24, Column, 4);
        multPrune<RDFINTINT>(G, dm_24, m_64);
        double t8_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 8 (Total) : " << (t8_end - t8_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // ==> step 9
        if (myrank == 0) {
            cout << "step 9 : m_(6,4) = m_(3,5).T().D() x m_(6,4)" << endl;
        }
        double t9_start = MPI_Wtime();
        diagonalizeM(m_35, dm_35_1, Column);
        multPrune<PTINTINT>(dm_35_1, m_64, m_64);
        double t9_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 9 (Total) : " << (t9_end - t9_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // ==> step 10
        if (myrank == 0) {
            cout << "step 10 : m_(3,5) = m_(3,5) x m_(6,4).D()" << endl;
        }
        double t10_start = MPI_Wtime();
        diagonalizeM(m_64, dm_64);
        multPrune<PTINTINT>(m_35, dm_64, m_35);
        double t10_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 10 (Total) : " << (t10_end - t10_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // ==> step 11
        if (myrank == 0) {
            cout << "step 11 : m_(4,3) = m_(6,4).T().D() x m_(4,3)" << endl;
        }
        double t11_start = MPI_Wtime();
        diagonalizeM(m_64, dm_64_1, Column);
        multPrune<PTINTINT>(dm_64_1, m_43, m_43);
        double t11_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 11 (Total) : " << (t11_end - t11_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // ==> step 12
        if (myrank == 0) {
            cout << "step 12 : m_(3,5) = m_(4,3).T().D() x m_(3,5)" << endl;
        }
        double t12_start = MPI_Wtime();
        diagonalizeM(m_43, dm_43_1, Column);
        multPrune<PTINTINT>(dm_43_1, m_35, m_35);
        double t12_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 12 (Total) : " << (t12_end - t12_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // ==> step 13
        if (myrank == 0) {
            cout << "step 13 : m_(5,0) = m_(3,5).T().D() x m_(5,0)" << endl;
        }
        double t13_start = MPI_Wtime();
        diagonalizeM(m_35, dm_35_2, Column);
        multPrune<PTINTINT>(dm_35_2, m_50, m_50);
        double t13_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 13 (Total) : " << (t13_end - t13_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // end count time
        double total_computing_2 = MPI_Wtime();

        printReducedInfo(m_50);

        if (myrank == 0) {
            cout << "total mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
            cout << "total prune time : " << total_prune_time << " s" << endl;
            cout << "total reduce time : " << total_reduce_time << " s" << endl;
            cout << "total cons_diag time : " << total_construct_diag_time << " s" << endl;
            cout << "total mult time : " << total_mult_time << " s" << endl;
            cout << "query7 totally takes : " << total_computing_2 - total_computing_1 << " s" << endl;
        }

    } else {            // multDimApplyPrune

        FullyDistVec<IndexType, ElementType> dm_50(commWorld), dm_35(commWorld), dm_13(commWorld),
            dm_43(commWorld), dm_24(commWorld), dm_35_1(commWorld), dm_64(commWorld), dm_64_1(commWorld),
            dm_43_1(commWorld), dm_35_2(commWorld);

        auto m_50(G), m_35(G), m_13(tG), m_43(tG), m_24(tG), m_64(G);

        FullyDistVec<IndexType, ElementType> r_50(commWorld, G.getnrow(), 0), l_13(commWorld, G.getnrow(), 0), l_24(commWorld, G.getnrow(), 0);
        r_50.SetElement(ind1, 6);
        l_13.SetElement(ind2, 1);
        l_24.SetElement(ind3, 1);

        // start count time
        double total_computing_1 = MPI_Wtime();

        // ==> step 1
        if (myrank == 0) {
            cout << "\n###############################################################" << endl;
            cout << "Query 7" << endl;
            cout << "###############################################################" << endl;
            cout << "---------------------------------------------------------------" << endl;
            cout << "step 1 : m_(5,0) = G x {1@(1345,1345)}*6" << endl;
        }
        double t1_start = MPI_Wtime();
        multDimApplyPrune(m_50, r_50, Column, true);
        double t1_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 1 (Total) : " << (t1_end - t1_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // ==> step 2
        if (myrank == 0) {
            cout << "step 2 : m_(3, 5) = G x m_(5, 0).D()*13" << endl;
        }
        double t2_start = MPI_Wtime();
        diagonalizeV(m_50, dm_50, Row, 13);
        multDimApplyPrune(m_35, dm_50, Column, true);
        double t2_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 2 (Total) : " << (t2_end - t2_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // ==> step 3
        if (myrank == 0) {
            cout << "step 3 : m_(1,3) = G.T() x m_(3,5).D()*6" << endl;
        }
        double t3_start = MPI_Wtime();
        diagonalizeV(m_35, dm_35, Row, 6);
        multDimApplyPrune(m_13, dm_35, Column, true);
        double t3_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 3 (Total) : " << (t3_end - t3_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // ==> step 4
        if (myrank == 0) {
            cout << "step 4 : m_(1,3) = {1@(43,43)} x m_(1,3)" << endl;
        }
        double t4_start = MPI_Wtime();
        multDimApplyPrune(m_13, l_13, Row, false);
        double t4_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 4 (Total) : " << (t4_end - t4_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // ==> step 5
        if (myrank == 0) {
            cout << "step 5 : m_(4,3) = G.T() x m_(1,3).T().D()*8" << endl;
        }
        double t5_start = MPI_Wtime();
        diagonalizeV(m_13, dm_13, Column, 8);
        multDimApplyPrune(m_43, dm_13, Column, true);
        double t5_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 5 (Total) : " << (t5_end - t5_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // ==> step 6
        if (myrank == 0) {
            cout << "step 6 : m_(2,4) = G.T() x m_(4,3).D()*6" << endl;
        }
        double t6_start = MPI_Wtime();
        diagonalizeV(m_43, dm_43, Row, 6);
        multDimApplyPrune(m_24, dm_43, Column, true);
        double t6_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 6 (Total) : " << (t6_end - t6_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // ==> step 7
        if (myrank == 0) {
            cout << "step 7 : m_(2,4) = {1@(79,79)} x m_(2,4)" << endl;
        }
        double t7_start = MPI_Wtime();
        multDimApplyPrune(m_24, l_24, Row, false);
        double t7_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 7 (Total) : " << (t7_end - t7_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // ==> step 8
        if (myrank == 0) {
            cout << "step 8 : m_(6,4) = G x m_(2,4).T().D()*4" << endl;
        }
        double t8_start = MPI_Wtime();
        diagonalizeV(m_24, dm_24, Column, 4);
        multDimApplyPrune(m_64, dm_24, Column, true);
        double t8_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 8 (Total) : " << (t8_end - t8_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // ==> step 9
        if (myrank == 0) {
            cout << "step 9 : m_(6,4) = m_(3,5).T().D() x m_(6,4)" << endl;
        }
        double t9_start = MPI_Wtime();
        diagonalizeV(m_35, dm_35_1, Column);
        multDimApplyPrune(m_64, dm_35_1, Row, false);
        double t9_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 9 (Total) : " << (t9_end - t9_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // ==> step 10
        if (myrank == 0) {
            cout << "step 10 : m_(3,5) = m_(3,5) x m_(6,4).D()" << endl;
        }
        double t10_start = MPI_Wtime();
        diagonalizeV(m_64, dm_64);
        multDimApplyPrune(m_35, dm_64, Column, false);
        double t10_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 10 (Total) : " << (t10_end - t10_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // ==> step 11
        if (myrank == 0) {
            cout << "step 11 : m_(4,3) = m_(6,4).T().D() x m_(4,3)" << endl;
        }
        double t11_start = MPI_Wtime();
        diagonalizeV(m_64, dm_64_1, Column);
        multDimApplyPrune(m_43, dm_64_1, Row, false);
        double t11_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 11 (Total) : " << (t11_end - t11_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // ==> step 12
        if (myrank == 0) {
            cout << "step 12 : m_(3,5) = m_(4,3).T().D() x m_(3,5)" << endl;
        }
        double t12_start = MPI_Wtime();
        diagonalizeV(m_43, dm_43_1, Column);
        multDimApplyPrune(m_35, dm_43_1, Row, false);
        double t12_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 12 (Total) : " << (t12_end - t12_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // ==> step 13
        if (myrank == 0) {
            cout << "step 13 : m_(5,0) = m_(3,5).T().D() x m_(5,0)" << endl;
        }
        double t13_start = MPI_Wtime();
        diagonalizeV(m_35, dm_35_2, Column);
        multDimApplyPrune(m_50, dm_35_2, Row, false);
        double t13_end = MPI_Wtime();

        if (myrank == 0) {
            cout << "step 13 (Total) : " << (t13_end - t13_start) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // end count time
        double total_computing_2 = MPI_Wtime();

        printReducedInfo(m_50);

        if (myrank == 0) {
            cout << "query7 mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
            cout << "query7 prune time : " << total_prune_time << " s" << endl;
            cout << "query7 diag_reduce time : " << total_reduce_time << " s" << endl;
            cout << "query7 dim_apply time : " << total_dim_apply_time << " s" << endl;
            cout << "query7 time (Total) : " << total_computing_2 - total_computing_1 << " s" << endl;
        }
    }
}

int main(int argc, char *argv[]) {
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 1) {
        if (myrank == 0) {
            cout << "Usage: ./lubm10240" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    {
        // initialization phase
        MPI_Barrier(MPI_COMM_WORLD);

        if (myrank == 0) {
            cout << "###############################################################" << endl;
            cout << "Load Matrix" << endl;
            cout << "###############################################################" << endl;
            cout << "---------------------------------------------------------------" << endl;
            cout << "starting reading lubm10240 data......" << endl;
        }

        double t_pre1 = MPI_Wtime();

        string Mname("/home/cheny0l/work/db245/fuad/data/lubm10240/encoded.mm");
//        string Mname("/project/k1285/fuad/data/lubm10240/encoded.mm");

        double t1 = MPI_Wtime();
        PSpMat::MPI_DCCols G(MPI_COMM_WORLD);
        G.ParallelReadMM(Mname, true, selectSecond);
        double t2 = MPI_Wtime();

        G.PrintInfo();

        if (myrank == 0) {
            cout << "\tread file takes : " << (t2 - t1) << " s" << endl;
        }

        auto commWorld = G.getcommgrid();

        float imG = G.LoadImbalance();
        if (myrank == 0) {
            cout << "\toriginal imbalance of G : " << imG << endl;
        }

        FullyDistVec<IndexType, IndexType> nonisov;
        permute(G, nonisov);

        double t1_trans = MPI_Wtime();
        auto tG = transpose(G);
        double t2_trans = MPI_Wtime();

        if (myrank == 0) {
            cout << "\ttranspose G takes : " << (t2_trans - t1_trans) << " s" << endl;
            cout << "graph load (Total) : " << (t2_trans - t_pre1) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        rvec = new FullyDistVec<IndexType, ElementType>(commWorld);
        rvec->iota(G.getnrow(), 0);
        qvec = new FullyDistVec<IndexType, ElementType>(commWorld);
        qvec->iota(G.getnrow(), 0);

        // query
        for (int time = 1; time <= 5; time++) {
            lubm10240_l1(G, tG, nonisov);
            lubm10240_l2(G, tG, nonisov);
            lubm10240_l3(G, tG, nonisov);
            lubm10240_l4(G, tG, nonisov);
            lubm10240_l5(G, tG, nonisov);
            lubm10240_l6(G, tG, nonisov);
            lubm10240_l7(G, tG, nonisov);
        }
    }

    MPI_Finalize();
    return 0;
}
