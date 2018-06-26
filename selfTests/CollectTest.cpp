#include <iostream>
#include <functional>
#include <algorithm>
#include <sstream>
#include "../include/CombBLAS.h"

#define ElementType int
#define IndexType uint32_t

using namespace std;

using namespace combblas;

template<class NT>
class PSpMat {
public:
    typedef SpDCCols<IndexType, NT> DCCols;
    typedef SpParMat<IndexType, NT, DCCols> MPI_DCCols;
};

// all three fully distributed vectors should already have same size, size of nnz
template<class DER>
void gatherMatrix(SpParMat<IndexType, ElementType, DER> &M, FullyDistVec<IndexType, ElementType> &ri,
                  FullyDistVec<IndexType, ElementType> &ci, FullyDistVec<IndexType, ElementType> &vi) {
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
                        vi.SetElement(index, ents[k].second);
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

int main(int argc, char *argv[]) {
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 2) {
        if (myrank == 0) {
            cout << "Usage: ./CollectTest <Matrix>" << endl;
            cout << "<Matrix> is file address, and file should be in triples format" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    {
        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));

        string Mname(argv[1]);

        MPI_Barrier(MPI_COMM_WORLD);

        PSpMat<ElementType>::MPI_DCCols A(MPI_COMM_WORLD);

        double t1 = MPI_Wtime();
        A.ReadDistribute(Mname, 0);
//        A.ParallelReadMM(Mname, true, maximum<ElementType>());
        double t2 = MPI_Wtime();
        if (myrank == 0) {
            cout << "read file takes " << t2 - t1 << " s" << endl;
        }
        A.PrintInfo();

        FullyDistVec<IndexType, ElementType> ri(A.getcommgrid(), A.getnnz(), 0);
        FullyDistVec<IndexType, ElementType> ci(A.getcommgrid(), A.getnnz(), 0);
        FullyDistVec<IndexType, ElementType> vi(A.getcommgrid(), A.getnnz(), 0);

        gatherMatrix(A, ri, ci, vi);

        ri.PrintInfo("row index");
        ri.ParallelWrite("/home/cheny0l/work/db245/CombBLAS_beta_16_1/build/row.index", false);

    }

    MPI_Finalize();
    return 0;
}
