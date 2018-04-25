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

ElementType my_or(ElementType a, ElementType b) {
    if (a != 0 &&  b != 0 && a == b) {
        return static_cast<ElementType>(1);
    } else {
        return static_cast<ElementType>(0);
    }
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

static double total_dim_apply_time = 0.0;

ElementType rdf_multiply(ElementType a, ElementType b) {
    if (a != 0 &&  b != 0 && a == b) {
        return static_cast<ElementType>(1);
    } else {
        return static_cast<ElementType>(0);
    }
}

void multDimApplyPrune(int step, PSpMat::MPI_DCCols &A, FullyDistVec<IndexType, ElementType> &v, Dim dim, bool isRDF) {
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
        cout << "    dim-apply takes: " << (t2 - t1) << " s" << endl;
    }

    double t3 = MPI_Wtime();
    A.Prune(isZero);
    double t4 = MPI_Wtime();

    if (myrank == 0) {
        total_prune_time += (t4 - t3);
        cout << "    prune takes: " << (t4 - t3) << " s         --- end of step " << step << "\n" << endl;
    }

    printReducedInfo(A);
}

void lubm320_L7(PSpMat::MPI_DCCols &G) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    FullyDistVec<IndexType, ElementType> nonisov(G.getcommgrid());
    permute(G, nonisov);
//    nonisov.ParallelWrite("nonisov.txt", false);

    double t1_trans = MPI_Wtime();
    auto tG = transpose(G);
    double t2_trans = MPI_Wtime();

    float imtpG = G.LoadImbalance();
    if (myrank == 0) {
        cout << "    transpose G takes : " << (t2_trans - t1_trans) << " s" <<endl;
        cout << "    imbalance of tG : " << imtpG << "\n" << endl;
    }

    double t_cons1 = MPI_Wtime();

    IndexType ind1 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(107)))[0];
    IndexType ind2 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(124)))[0];
    IndexType ind3 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(2079)))[0];

    int nrow = G.getnrow(), ncol = G.getncol();

    std::vector<int> riv1(1, ind2);
    std::vector<int> viv1(1, 1);

    FullyDistVec<int, ElementType> ri1(riv1, G.getcommgrid());
    FullyDistVec<int, ElementType> vi1(viv1, G.getcommgrid());

    PSpMat::MPI_DCCols l_14(nrow, ncol, ri1, ri1, vi1);


    std::vector<int> riv2(1, ind3);
    std::vector<int> viv2(1, 1);

    FullyDistVec<int, ElementType> ri2(riv2, G.getcommgrid());
    FullyDistVec<int, ElementType> vi2(viv2, G.getcommgrid());

    PSpMat::MPI_DCCols l_25(nrow, ncol, ri2, ri2, vi2);

    double t_cons2 = MPI_Wtime();
    if (myrank == 0) {
        cout << "    construct single element matrix takes : " << (t_cons2 - t_cons1) << "\n" << endl;
    }

    FullyDistVec<IndexType, ElementType> r_30(G.getnrow(), 0);
    r_30.SetElement(ind1, 8);
    r_30.ParallelWrite("r_30", false);

    // start count time
    double total_computing_1 = MPI_Wtime();

    // ==> step 1
    auto m_30(G);
    multDimApplyPrune(1, m_30, r_30, Column, true);

    // ==> step 2
    FullyDistVec<int, ElementType> diag(G.getcommgrid());
    m_30.Reduce(diag, Row, std::logical_or<ElementType>() , 0);
    diag.Apply(bind2nd(multiplies<ElementType>(), 2));

    double t_dim1 = MPI_Wtime();
    PSpMat::MPI_DCCols m_43(tG);
    m_43.DimApply(Column, diag, my_or);
    m_43.Prune(isZero);
    m_43.PrintInfo();
    double t_dim2 = MPI_Wtime();
    if (myrank == 0) {
        cout << "    dim apply : " << (t_dim2 - t_dim1) << " s" << endl;
    }

    auto dm_43 = diagonalize(m_43);
    mmul_scalar(dm_43, 8);

    // ==> step 3
    PSpMat::MPI_DCCols m_14(MPI_COMM_WORLD);
    multPrune<RDFINTINT>(tG, dm_43, m_14);

    // ==> step 4
    multPrune<PTINTINT>(l_14, m_14, m_14);

    auto dm_14 = diagonalize(m_14, true);
    mmul_scalar(dm_14, 11);

    // ==> step 5
    PSpMat::MPI_DCCols m_54(MPI_COMM_WORLD);
    multPrune<RDFINTINT>(G, dm_14, m_54);

    auto dm_54 = diagonalize(m_54);
    mmul_scalar(dm_54, 8);

    // ==> step 6
    PSpMat::MPI_DCCols m_25(MPI_COMM_WORLD);
    multPrune<RDFINTINT>(tG, dm_54, m_25);

    // ==> step 7
    multPrune<PTINTINT>(l_25, m_25, m_25);

    auto dm_25 = diagonalize(m_25, true);
    mmul_scalar(dm_25, 7);

    // ==> step 8
    PSpMat::MPI_DCCols m_65(MPI_COMM_WORLD);
    multPrune<RDFINTINT>(G, dm_25, m_65);

    auto dm_43_1 = diagonalize(m_43, true);

    // ==> step 9
    multPrune<PTINTINT>(dm_43_1, m_65, m_65);

    auto dm_65 = diagonalize(m_65);

    // ==> step 10
    multPrune<PTINTINT>(m_43, dm_65, m_43);

    auto dm_65_1 = diagonalize(m_65, true);

    // ==> step 11
    multPrune<PTINTINT>(dm_65_1, m_54, m_54);

    auto dm_54_1 = diagonalize(m_54, true);

    // ==> step 12
    multPrune<PTINTINT>(dm_54_1, m_43, m_43);

    auto dm_43_2 = diagonalize(m_43, true);

    // ==> step 13
    multPrune<PTINTINT>(dm_43_2, m_30, m_30);

    // end count time
    double total_computing_2 = MPI_Wtime();

    printReducedInfo(m_30);

    if(myrank == 0) {
        cout << "total mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "total transpose time : " << total_transpose_time << " s" << endl;
        cout << "total prune time : " << total_prune_time << " s" << endl;
        cout << "total reduce time : " << total_reduce_time << " s" << endl;
        cout << "total cons_diag time : " << total_construct_diag_time << " s" << endl;
        cout << "total mult time : " << total_mult_time << " s" << endl;
        cout << "query7 totally takes : " << total_computing_2 - total_computing_1 << " s" << endl;
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
        PSpMat::MPI_DCCols G(MPI_COMM_WORLD);

        double t1 = MPI_Wtime();

        G.ParallelReadMM(Mname, true, maximum<ElementType>());
        G.PrintInfo();

        double t2 = MPI_Wtime();

        rvec = new FullyDistVec<int, int>(fullWorld);
        rvec->iota(G.getnrow(), 0);
        qvec = new FullyDistVec<int, int>(fullWorld);
        qvec->iota(G.getnrow(), 0);

        float imG = G.LoadImbalance();
        if (myrank == 0) {
            cout << "read file takes : " << (t2 - t1) << " s" << endl;
            cout << "imbalcance of G : " << imG << endl;
        }

        lubm320_L7(G);
    }

    MPI_Finalize();
    return 0;
}
