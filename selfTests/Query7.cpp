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
        std::vector<int> riv(1, 107);
        std::vector<int> civ(1, 107);
        std::vector<int> viv(1, 8);

        FullyDistVec<int, ElementType> ri(riv, fullWorld);
        FullyDistVec<int, ElementType> ci(civ, fullWorld);
        FullyDistVec<int, ElementType> vi(viv, fullWorld);

        PSpMat<ElementType>::MPI_DCCols r_30(nrow, ncol, ri, ci, vi);

        // ==> step 1
        auto m_30 = Mult_AnXBn_DoubleBuff<RDFINTINT, ElementType, PSpMat<ElementType>::DCCols>(G, r_30);
        m_30.Prune(isZero);

        int nnz1 = m_30.getnnz();
        if (myrank == 0) {
            cout << "m_(3, 0) : " << nnz1;
        }
        FullyDistVec<int, ElementType> rowsums1(fullWorld);
        m_30.Reduce(rowsums1, Row, std::plus<ElementType>() , 0);
        FullyDistVec<int, ElementType> colsums1(fullWorld);
        m_30.Reduce(colsums1, Column, std::plus<ElementType>() , 0);
        int nnzrows1 = rowsums1.Count(isNotZero);
        int nnzcols1 = colsums1.Count(isNotZero);
        if (myrank == 0) {
            cout << " [ " << nnzrows1 << ", " << nnzcols1 << " ]" << endl;
        }

        auto tG = transpose(G);
        auto dm_30 = diagonalize(m_30);
        mmul_scalar(dm_30, 2);

        // ==> step 2
        auto m_43 = Mult_AnXBn_DoubleBuff<RDFINTINT, ElementType, PSpMat<ElementType>::DCCols>(tG, dm_30);
        m_43.Prune(isZero);

        int nnz2 = m_43.getnnz();
        if (myrank == 0) {
            cout << "m_(4, 3) : " << nnz2;
        }
        FullyDistVec<int, ElementType> rowsums2(fullWorld);
        m_43.Reduce(rowsums2, Row, std::plus<ElementType>() , 0);
        FullyDistVec<int, ElementType> colsums2(fullWorld);
        m_43.Reduce(colsums2, Column, std::plus<ElementType>() , 0);
        int nnzrows2 = rowsums2.Count(isNotZero);
        int nnzcols2 = colsums2.Count(isNotZero);
        if (myrank == 0) {
            cout << " [ " << nnzrows2 << ", " << nnzcols2 << " ]" << endl;
        }

        auto dm_43 = diagonalize(m_43);
        mmul_scalar(dm_43, 8);

        // ==> step 3
        auto m_14 = Mult_AnXBn_DoubleBuff<RDFINTINT, ElementType, PSpMat<ElementType>::DCCols>(tG, dm_43);
        m_14.Prune(isZero);

        int nnz3 = m_14.getnnz();
        if (myrank == 0) {
            cout << "m_(1, 4) : " << nnz3;
        }
        FullyDistVec<int, ElementType> rowsums3(fullWorld);
        m_14.Reduce(rowsums3, Row, std::plus<ElementType>() , 0);
        FullyDistVec<int, ElementType> colsums3(fullWorld);
        m_14.Reduce(colsums3, Column, std::plus<ElementType>() , 0);
        int nnzrows3 = rowsums3.Count(isNotZero);
        int nnzcols3 = colsums3.Count(isNotZero);
        if (myrank == 0) {
            cout << " [ " << nnzrows3 << ", " << nnzcols3 << " ]" << endl;
        }

        int nrow1 = m_14.getnrow(), ncol1 = m_14.getncol();
        std::vector<int> riv1(1, 124);
        std::vector<int> civ1(1, 124);
        std::vector<int> viv1(1, 1);

        FullyDistVec<int, ElementType> ri1(riv1, fullWorld);
        FullyDistVec<int, ElementType> ci1(civ1, fullWorld);
        FullyDistVec<int, ElementType> vi1(viv1, fullWorld);

        PSpMat<ElementType>::MPI_DCCols l_14(nrow1, ncol1, ri1, ci1, vi1);

        // ==> step 4
        m_14 = Mult_AnXBn_DoubleBuff<PTINTINT, ElementType, PSpMat<ElementType>::DCCols>(l_14, m_14);
        m_14.Prune(isZero);

        int nnz4 = m_14.getnnz();
        if (myrank == 0) {
            cout << "m_(1, 4) : " << nnz4;
        }
        FullyDistVec<int, ElementType> rowsums4(fullWorld);
        m_14.Reduce(rowsums4, Row, std::plus<ElementType>() , 0);
        FullyDistVec<int, ElementType> colsums4(fullWorld);
        m_14.Reduce(colsums4, Column, std::plus<ElementType>() , 0);
        int nnzrows4 = rowsums4.Count(isNotZero);
        int nnzcols4 = colsums4.Count(isNotZero);
        if (myrank == 0) {
            cout << " [ " << nnzrows4 << ", " << nnzcols4 << " ]" << endl;
        }

        auto tm_14 = transpose(m_14);
        auto dm_14 = diagonalize(tm_14);
        mmul_scalar(dm_14, 11);

        // ==> step 5
        auto m_54 = Mult_AnXBn_DoubleBuff<RDFINTINT, ElementType, PSpMat<ElementType>::DCCols>(G, dm_14);
        m_54.Prune(isZero);

        int nnz5 = m_54.getnnz();
        if (myrank == 0) {
            cout << "m_(5, 4) : " << nnz5;
        }
        FullyDistVec<int, ElementType> rowsums5(fullWorld);
        m_54.Reduce(rowsums5, Row, std::plus<ElementType>() , 0);
        FullyDistVec<int, ElementType> colsums5(fullWorld);
        m_54.Reduce(colsums5, Column, std::plus<ElementType>() , 0);
        int nnzrows5 = rowsums5.Count(isNotZero);
        int nnzcols5 = colsums5.Count(isNotZero);
        if (myrank == 0) {
            cout << " [ " << nnzrows5 << ", " << nnzcols5 << " ]" << endl;
        }

        auto dm_54 = diagonalize(m_54);
        mmul_scalar(dm_54, 8);

        // ==> step 6
        auto m_25 = Mult_AnXBn_DoubleBuff<RDFINTINT, ElementType, PSpMat<ElementType>::DCCols>(tG, dm_54);
        m_25.Prune(isZero);

        int nnz6 = m_25.getnnz();
        if (myrank == 0) {
            cout << "m_(2, 5) : " << nnz6;
        }
        FullyDistVec<int, ElementType> rowsums6(fullWorld);
        m_25.Reduce(rowsums6, Row, std::plus<ElementType>() , 0);
        FullyDistVec<int, ElementType> colsums6(fullWorld);
        m_25.Reduce(colsums6, Column, std::plus<ElementType>() , 0);
        int nnzrows6 = rowsums6.Count(isNotZero);
        int nnzcols6 = colsums6.Count(isNotZero);
        if (myrank == 0) {
            cout << " [ " << nnzrows6 << ", " << nnzcols6 << " ]" << endl;
        }

        int nrow2 = m_25.getnrow(), ncol2 = m_25.getncol();
        std::vector<int> riv2(1, 2079);
        std::vector<int> civ2(1, 2079);
        std::vector<int> viv2(1, 1);

        FullyDistVec<int, ElementType> ri2(riv2, fullWorld);
        FullyDistVec<int, ElementType> ci2(civ2, fullWorld);
        FullyDistVec<int, ElementType> vi2(viv2, fullWorld);

        PSpMat<ElementType>::MPI_DCCols l_25(nrow2, ncol2, ri2, ci2, vi2);

        // ==> step 7
        m_25 = Mult_AnXBn_DoubleBuff<PTINTINT, ElementType, PSpMat<ElementType>::DCCols>(l_25, m_25);
        m_25.Prune(isZero);

        int nnz7 = m_25.getnnz();
        if (myrank == 0) {
            cout << "m_(2, 5) : " << nnz7;
        }
        FullyDistVec<int, ElementType> rowsums7(fullWorld);
        m_25.Reduce(rowsums7, Row, std::plus<ElementType>() , 0);
        FullyDistVec<int, ElementType> colsums7(fullWorld);
        m_25.Reduce(colsums7, Column, std::plus<ElementType>() , 0);
        int nnzrows7 = rowsums7.Count(isNotZero);
        int nnzcols7 = colsums7.Count(isNotZero);
        if (myrank == 0) {
            cout << " [ " << nnzrows7 << ", " << nnzcols7 << " ]" << endl;
        }

        auto tm_25 = transpose(m_25);
        auto dm_25 = diagonalize(tm_25);
        mmul_scalar(dm_25, 7);

        // ==> step 8
        auto m_65 = Mult_AnXBn_DoubleBuff<RDFINTINT, ElementType, PSpMat<ElementType>::DCCols>(G, dm_25);
        m_65.Prune(isZero);

        int nnz8 = m_65.getnnz();
        if (myrank == 0) {
            cout << "m_(6, 5) : " << nnz8;
        }
        FullyDistVec<int, ElementType> rowsums8(fullWorld);
        m_65.Reduce(rowsums8, Row, std::plus<ElementType>() , 0);
        FullyDistVec<int, ElementType> colsums8(fullWorld);
        m_65.Reduce(colsums8, Column, std::plus<ElementType>() , 0);
        int nnzrows8 = rowsums8.Count(isNotZero);
        int nnzcols8 = colsums8.Count(isNotZero);
        if (myrank == 0) {
            cout << " [ " << nnzrows8 << ", " << nnzcols8 << " ]" << endl;
        }

        auto tm_43 = transpose(m_43);
        auto dm_43_1 = diagonalize(tm_43);

        // ==> step 9
        m_65 = Mult_AnXBn_DoubleBuff<PTINTINT, ElementType, PSpMat<ElementType>::DCCols>(dm_43_1, m_65);
        m_65.Prune(isZero);

        int nnz9 = m_65.getnnz();
        if (myrank == 0) {
            cout << "m_(6, 5) : " << nnz9;
        }
        FullyDistVec<int, ElementType> rowsums9(fullWorld);
        m_65.Reduce(rowsums9, Row, std::plus<ElementType>() , 0);
        FullyDistVec<int, ElementType> colsums9(fullWorld);
        m_65.Reduce(colsums9, Column, std::plus<ElementType>() , 0);
        int nnzrows9 = rowsums9.Count(isNotZero);
        int nnzcols9 = colsums9.Count(isNotZero);
        if (myrank == 0) {
            cout << " [ " << nnzrows9 << ", " << nnzcols9 << " ]" << endl;
        }

        auto dm_65 = diagonalize(m_65);

        // ==> step 10
        m_43 = Mult_AnXBn_DoubleBuff<PTINTINT, ElementType, PSpMat<ElementType>::DCCols>(m_43, dm_65);
        m_43.Prune(isZero);

        int nnz10 = m_43.getnnz();
        if (myrank == 0) {
            cout << "m_(4, 3) : " << nnz10;
        }
        FullyDistVec<int, ElementType> rowsums10(fullWorld);
        m_43.Reduce(rowsums10, Row, std::plus<ElementType>() , 0);
        FullyDistVec<int, ElementType> colsums10(fullWorld);
        m_43.Reduce(colsums10, Column, std::plus<ElementType>() , 0);
        int nnzrows10 = rowsums10.Count(isNotZero);
        int nnzcols10 = colsums10.Count(isNotZero);
        if (myrank == 0) {
            cout << " [ " << nnzrows10 << ", " << nnzcols10 << " ]" << endl;
        }

        auto tm_65 = transpose(m_65);
        auto dm_65_1 = diagonalize(tm_65);

        // ==> step 11
        m_54 = Mult_AnXBn_DoubleBuff<PTINTINT, ElementType, PSpMat<ElementType>::DCCols>(dm_65_1, m_54);
        m_54.Prune(isZero);

        int nnz11 = m_54.getnnz();
        if (myrank == 0) {
            cout << "m_(5, 4) : " << nnz11;
        }
        FullyDistVec<int, ElementType> rowsums11(fullWorld);
        m_54.Reduce(rowsums11, Row, std::plus<ElementType>() , 0);
        FullyDistVec<int, ElementType> colsums11(fullWorld);
        m_54.Reduce(colsums11, Column, std::plus<ElementType>() , 0);
        int nnzrows11 = rowsums11.Count(isNotZero);
        int nnzcols11 = colsums11.Count(isNotZero);
        if (myrank == 0) {
            cout << " [ " << nnzrows11 << ", " << nnzcols11 << " ]" << endl;
        }

        auto tm_54 = transpose(m_54);
        auto dm_54_1 = diagonalize(tm_54);

        // ==> step 12
        m_43 = Mult_AnXBn_DoubleBuff<PTINTINT, ElementType, PSpMat<ElementType>::DCCols>(dm_54_1, m_43);
        m_43.Prune(isZero);

        int nnz12 = m_43.getnnz();
        if (myrank == 0) {
            cout << "m_(4, 3) : " << nnz12;
        }
        FullyDistVec<int, ElementType> rowsums12(fullWorld);
        m_43.Reduce(rowsums12, Row, std::plus<ElementType>() , 0);
        FullyDistVec<int, ElementType> colsums12(fullWorld);
        m_43.Reduce(colsums12, Column, std::plus<ElementType>() , 0);
        int nnzrows12 = rowsums12.Count(isNotZero);
        int nnzcols12 = colsums12.Count(isNotZero);
        if (myrank == 0) {
            cout << " [ " << nnzrows12 << ", " << nnzcols12 << " ]" << endl;
        }

        auto tm_43_1 = transpose(m_43);
        auto dm_43_2 = diagonalize(tm_43_1);

        // ==> step 13
        m_30 = Mult_AnXBn_DoubleBuff<PTINTINT, ElementType, PSpMat<ElementType>::DCCols>(dm_43_2, m_30);
        m_30.Prune(isZero);

        int nnz13 = m_30.getnnz();
        if (myrank == 0) {
            cout << "m_(3, 0) : " << nnz13;
        }
        FullyDistVec<int, ElementType> rowsums13(fullWorld);
        m_30.Reduce(rowsums13, Row, std::plus<ElementType>() , 0);
        FullyDistVec<int, ElementType> colsums13(fullWorld);
        m_30.Reduce(colsums13, Column, std::plus<ElementType>() , 0);
        int nnzrows13 = rowsums13.Count(isNotZero);
        int nnzcols13 = colsums13.Count(isNotZero);
        if (myrank == 0) {
            cout << " [ " << nnzrows13 << ", " << nnzcols13 << " ]" << endl;
        }

        // end count time
        double t2 = MPI_Wtime();

        m_30.SaveGathered("m_30.txt");

        if(myrank == 0) {
            cout << "query 7 takes " << t2 - t1 << " s" << endl;
        }
    }

    MPI_Finalize();
    return 0;
}