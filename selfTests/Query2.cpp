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

void set_element(PSpMat<ElementType>::MPI_DCCols &M, int i, int j, ElementType v) {

    std::vector<int> riv(1, i);
    std::vector<int> civ(1, j);
    std::vector<ElementType> viv(1, v);

    FullyDistVec<int, ElementType> ri(riv, M.getcommgrid());
    FullyDistVec<int, ElementType> ci(civ, M.getcommgrid());
    FullyDistVec<int, ElementType> vi(viv, M.getcommgrid());

    ri.Apply(bind2nd(minus<int>(), 1));
    ci.Apply(bind2nd(minus<int>(), 1));

    PSpMat<ElementType>::MPI_DCCols B(M.getnrow(), M.getncol(), ri, ci, vi);
//    std::cout << "B : ";
//    B.PrintInfo();

    M.Prune(ri, ci);
    M += B;
}

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
//        string Mname("/home/cheny0l/work/db245/CombBLAS_beta_16_1/TESTDATA/gen_10_10_25.txt");
        PSpMat<ElementType>::MPI_DCCols G(MPI_COMM_WORLD);

        G.ReadDistribute(Mname, 0);
        G.PrintInfo();

        int nrow = G.getnrow(), ncol = G.getncol();
        std::vector<int> riv(1, 124);
        std::vector<int> civ(1, 124);
//    std::vector<int> riv(1, 5);
//    std::vector<int> civ(1, 5);
        std::vector<int> viv(1, 8);

        FullyDistVec<int, ElementType> ri(riv, G.getcommgrid());
        FullyDistVec<int, ElementType> ci(riv, G.getcommgrid());
        FullyDistVec<int, ElementType> vi(riv, G.getcommgrid());

        PSpMat<ElementType>::MPI_DCCols r_10(nrow, ncol, ri, ci, vi);

        r_10.PrintInfo();

        auto m_10 = Mult_AnXBn_DoubleBuff<RDFINTINT, ElementType, PSpMat<ElementType>::DCCols>(G, r_10);

        m_10.SaveGathered("m_10.txt");
        m_10.Prune(isZero);

        int nnz1 = m_10.getnnz();
        if (myrank == 0) {
            cout << "m_(1, 0) : " << nnz1 << endl;
        }
//
//        auto tG = transpose(G);
//        auto dm_10 = diagonalize(m_10);
////        int nnz_dm_10 = dm_10.getnnz();
////        if (myrank == 0) {
////            cout << "dm_(1, 0) : " << nnz_dm_10 << endl;
////        }
//
//        mmul_scalar(dm_10, 3);
//        auto m_21 = Mult_AnXBn_DoubleBuff<RDFINTINT, ElementType, PSpMat<ElementType>::DCCols>(tG, dm_10, false, false);
//
//        m_21.Prune(isZero);
//        int nnz2 = m_21.getnnz();
//        if (myrank == 0) {
//            cout << "m_(2, 1) : " << nnz2 << endl;
//        }
//
//        auto dm_21 = transpose(m_21);
//        diagonalize(dm_21);
//
//        m_10 = Mult_AnXBn_DoubleBuff<PTINTINT, ElementType, PSpMat<ElementType>::DCCols>(dm_21, m_10, false, false);
//        m_10.Prune(isZero);
//
//        int nnz3 = m_10.getnnz();
//        if (myrank == 0) {
//            cout << "m_(1, 0) : " << nnz3 << endl;
//        }
    }

    MPI_Finalize();
    return 0;
}