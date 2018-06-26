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

// all three fully distributed vectors should already have same size, size of nnz
template<class DER>
void gatherMatrix(SpParMat<IndexType, ElementType, DER> &M, FullyDistVec<IndexType, IndexType> &ri,
                  FullyDistVec<IndexType, IndexType> &ci) {
    auto commGrid = M.getcommgrid();

    int proccols = commGrid->GetGridCols();
    int procrows = commGrid->GetGridRows();

    IndexType index = 0;

    int colrank = commGrid->GetRankInProcCol();
    int colneighs = commGrid->GetGridRows();
    IndexType *locnrows = new IndexType[colneighs];    // number of rows is calculated by a reduction among the processor column
    locnrows[colrank] = (IndexType) M.getlocalrows();
    MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IndexType>(), locnrows, 1, MPIType<IndexType>(), commGrid->GetColWorld());
    IndexType roffset = std::accumulate(locnrows, locnrows + colrank, 0);
    delete[] locnrows;

    MPI_Datatype datatype;
    MPI_Type_contiguous(sizeof(std::pair<IndexType, ElementType>), MPI_CHAR, &datatype);
    MPI_Type_commit(&datatype);

    for (int i = 0; i < procrows; i++)    // for all processor row (in order)
    {
        if (commGrid->GetRankInProcCol() == i)    // only the ith processor row
        {
            auto spSeq = M.seqptr();
            IndexType localrows = spSeq->getnrow();    // same along the processor row
            std::vector<std::vector<std::pair<IndexType, ElementType> > > csr(localrows);
            if (commGrid->GetRankInProcRow() == 0)    // get the head of processor row
            {
                IndexType localcols = spSeq->getncol();    // might be different on the last processor on this processor row
                MPI_Bcast(&localcols, 1, MPIType<IndexType>(), 0, commGrid->GetRowWorld());
                for (typename DER::SpColIter colit = spSeq->begcol();
                     colit != spSeq->endcol(); ++colit)    // iterate over nonempty subcolumns
                {
                    for (typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit);
                         nzit != spSeq->endnz(colit); ++nzit) {
                        csr[nzit.rowid()].push_back(std::make_pair(colit.colid(), nzit.value()));
                    }
                }
            } else    // get the rest of the processors
            {
                IndexType n_perproc;
                MPI_Bcast(&n_perproc, 1, MPIType<IndexType>(), 0, commGrid->GetRowWorld());
                IndexType noffset = commGrid->GetRankInProcRow() * n_perproc;
                for (typename DER::SpColIter colit = spSeq->begcol();
                     colit != spSeq->endcol(); ++colit)    // iterate over nonempty subcolumns
                {
                    for (typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit);
                         nzit != spSeq->endnz(colit); ++nzit) {
                        csr[nzit.rowid()].push_back(std::make_pair(colit.colid() + noffset, nzit.value()));
                    }
                }
            }
            std::pair<IndexType, ElementType> *ents = NULL;
            int *gsizes = NULL, *dpls = NULL;
            if (commGrid->GetRankInProcRow() == 0)    // only the head of processor row
            {
                gsizes = new int[proccols];
                dpls = new int[proccols]();    // displacements (zero initialized pid)
            }
            for (int j = 0; j < localrows; ++j) {
                IndexType rowcnt = 0;
                sort(csr[j].begin(), csr[j].end());
                int mysize = csr[j].size();
                MPI_Gather(&mysize, 1, MPI_INT, gsizes, 1, MPI_INT, 0, commGrid->GetRowWorld());
                if (commGrid->GetRankInProcRow() == 0) {
                    rowcnt = std::accumulate(gsizes, gsizes + proccols, static_cast<IndexType>(0));
                    std::partial_sum(gsizes, gsizes + proccols - 1, dpls + 1);
                    ents = new std::pair<IndexType, ElementType>[rowcnt];    // nonzero entries in the j'th local row
                }

                // int MPI_Gatherv (void* sbuf, int scount, MPI_Datatype stype,
                // 		    void* rbuf, int *rcount, int* displs, MPI_Datatype rtype, int root, MPI_Comm comm)
                MPI_Gatherv(SpHelper::p2a(csr[j]), mysize, datatype, ents, gsizes, dpls, datatype, 0,
                            commGrid->GetRowWorld());
                if (commGrid->GetRankInProcRow() == 0) {
                    for (int k = 0; k < rowcnt; ++k) {
                        ri.SetElement(index, j + roffset);
                        ci.SetElement(index, ents[k].first);
                        index++;
                    }
                    delete[] ents;
                }
            }
            if (commGrid->GetRankInProcRow() == 0) {
                DeleteAll(gsizes, dpls);
            }
        } // end_if the ith processor row
        MPI_Barrier(
                commGrid->GetWorld());        // signal the end of ith processor row iteration (so that all processors block)
    }
}

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

    FullyDistVec<int, ElementType> rowsums1(M.getcommgrid());
    M.Reduce(rowsums1, Row, std::plus<ElementType>(), 0);
    FullyDistVec<int, ElementType> colsums1(M.getcommgrid());
    M.Reduce(colsums1, Column, std::plus<ElementType>(), 0);
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
    FullyDistVec<IndexType, ElementType> *ColSums = new FullyDistVec<IndexType, ElementType>(G.getcommgrid());
    FullyDistVec<IndexType, ElementType> *RowSums = new FullyDistVec<IndexType, ElementType>(G.getcommgrid());
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

void
diagonalizeV(const PSpMat::MPI_DCCols &M, FullyDistVec<IndexType, ElementType> &diag, Dim dim = Row, int scalar = 1) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t1 = MPI_Wtime();
    M.Reduce(diag, dim, std::logical_or<ElementType>(), 0);
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

void resGen2(PSpMat::MPI_DCCols &m_10, PSpMat::MPI_DCCols &m_21) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    FullyDistVec<IndexType, IndexType> ri_10_1(m_10.getcommgrid(), m_10.getnnz(), 0);
    FullyDistVec<IndexType, IndexType> ci_10_0(m_10.getcommgrid(), m_10.getnnz(), 0);

    FullyDistVec<IndexType, IndexType> ri_21_2(m_21.getcommgrid(), m_21.getnnz(), 0);
    FullyDistVec<IndexType, IndexType> ci_21_1(m_21.getcommgrid(), m_21.getnnz(), 0);


    double t1_start = MPI_Wtime();
    gatherMatrix(m_10, ri_10_1, ci_10_0);
    gatherMatrix(m_21, ri_21_2, ci_21_1);
    double t1_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "\ngetting indices takes : " << (t1_end - t1_start) << " s" << endl;
    }

    // magic number 12000000
    FullyDistVec<IndexType, std::tuple<IndexType, IndexType, IndexType> > res(m_10.getcommgrid(), 12000000, std::tuple<IndexType, IndexType, IndexType>());
    int indRes = 0;

    double t2_start = MPI_Wtime();
    for (int i = 0; i < ri_10_1.glen; ++i) {
        if (myrank == 0 && i % 10000 == 0) {
            cout << "iteration : " << i << endl;
        }

        auto ni = ci_21_1.FindInds(bind2nd(equal_to<IndexType>(), ri_10_1[i]));

        for (int j = 0; j < ni.glen; ++j) {
            res.SetElement(indRes, std::tuple<IndexType, IndexType, IndexType>(ci_10_0[i], ri_10_1[i], ri_21_2[ni[j]]));
        }
    }
    double t2_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "join takes : " << (t2_end - t2_start) << " s\n" << endl;
    }

    res.PrintInfo("result");
//    res.ParallelWrite("/home/cheny0l/work/db245/CombBLAS_beta_16_1/build/res2.res", true);
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

    resGen2(m_10, m_21);
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

        FullyDistVec<IndexType, IndexType> nonisov(G.getcommgrid());
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
        lubm10240_l2(G, tG, nonisov);
    }

    MPI_Finalize();
    return 0;
}
