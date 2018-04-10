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
        PSpMat<ElementType>::MPI_DCCols G(MPI_COMM_WORLD);

        G.ReadDistribute(Mname, 0);
        G.PrintInfo();

        // start count time
        double t1 = MPI_Wtime();

        int nrow = G.getnrow(), ncol = G.getncol();
        std::vector<int> riv(1, 124);
        std::vector<int> civ(1, 124);
        std::vector<int> viv(1, 8);

        FullyDistVec<int, ElementType> ri(riv, fullWorld);
        FullyDistVec<int, ElementType> ci(civ, fullWorld);
        FullyDistVec<int, ElementType> vi(viv, fullWorld);

        PSpMat<ElementType>::MPI_DCCols r_10(nrow, ncol, ri, ci, vi);

        // ==> step 1
        auto m_10 = Mult_AnXBn_DoubleBuff<RDFINTINT, ElementType, PSpMat<ElementType>::DCCols>(G, r_10);
        m_10.Prune(isZero);

        int nnz1 = m_10.getnnz();
        if (myrank == 0) {
            cout << "m_(1, 0) : " << nnz1;
        }
        FullyDistVec<int, ElementType> rowsums1(fullWorld);
        m_10.Reduce(rowsums1, Row, std::plus<ElementType>() , 0);
        FullyDistVec<int, ElementType> colsums1(fullWorld);
        m_10.Reduce(colsums1, Column, std::plus<ElementType>() , 0);
        int nnzrows1 = rowsums1.Count(isNotZero);
        int nnzcols1 = colsums1.Count(isNotZero);
        if (myrank == 0) {
            cout << " [ " << nnzrows1 << ", " << nnzcols1 << " ]" << endl;
        }

        auto tG = transpose(G);
        auto dm_10 = diagonalize(m_10);
        mmul_scalar(dm_10, 3);

        // ==> step 2
        auto m_21 = Mult_AnXBn_DoubleBuff<RDFINTINT, ElementType, PSpMat<ElementType>::DCCols>(tG, dm_10);
        m_21.Prune(isZero);

        int nnz2 = m_21.getnnz();
        if (myrank == 0) {
            cout << "m_(2, 1) : " << nnz2;
        }
        FullyDistVec<int, ElementType> rowsums2(fullWorld);
        m_21.Reduce(rowsums2, Row, std::plus<ElementType>() , 0);
        FullyDistVec<int, ElementType> colsums2(fullWorld);
        m_21.Reduce(colsums2, Column, std::plus<ElementType>() , 0);
        int nnzrows2 = rowsums2.Count(isNotZero);
        int nnzcols2 = colsums2.Count(isNotZero);
        if (myrank == 0) {
            cout << " [ " << nnzrows2 << ", " << nnzcols2 << " ]" << endl;
        }

        auto tm_21 = transpose(m_21);
        auto dm_21 = diagonalize(tm_21);

        // ==> step 3
        m_10 = Mult_AnXBn_DoubleBuff<PTINTINT, ElementType, PSpMat<ElementType>::DCCols>(dm_21, m_10);
        m_10.Prune(isZero);

        int nnz3 = m_10.getnnz();
        if (myrank == 0) {
            cout << "m_(1, 0) : " << nnz3;
        }
        FullyDistVec<int, ElementType> rowsums3(fullWorld);
        m_10.Reduce(rowsums3, Row, std::plus<ElementType>() , 0);
        FullyDistVec<int, ElementType> colsums3(fullWorld);
        m_10.Reduce(colsums3, Column, std::plus<ElementType>() , 0);
        int nnzrows3 = rowsums3.Count(isNotZero);
        int nnzcols3 = colsums3.Count(isNotZero);
        if (myrank == 0) {
            cout << " [ " << nnzrows3 << ", " << nnzcols3 << " ]" << endl;
        }

        // end count time
        double t2 = MPI_Wtime();

        m_10.SaveGathered("m_10.txt");

        if(myrank == 0) {
            cout << "query 2 takes " << t2 - t1 << " s" << endl;
        }
    }

    MPI_Finalize();
    return 0;
}