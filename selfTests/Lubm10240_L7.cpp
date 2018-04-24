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

void lubm10240_L7(PSpMat::MPI_DCCols &G) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    FullyDistVec<IndexType, ElementType> nonisov(G.getcommgrid());
    permute(G, nonisov);

    double t1_trans = MPI_Wtime();
    auto tG = transpose(G);
    double t2_trans = MPI_Wtime();

    float imtpG = G.LoadImbalance();
    if (myrank == 0) {
        cout << "    transpose G takes : " << (t2_trans - t1_trans) << " s" <<endl;
        cout << "    imbalance of tG : " << imtpG << "\n" << endl;
    }

    double t_cons1 = MPI_Wtime();

    int nrow = G.getnrow(), ncol = G.getncol();
    std::vector<int> riv(1, 1345);
    std::vector<int> civ(1, 1345);
    std::vector<int> viv(1, 6);

    FullyDistVec<int, ElementType> ri(riv, G.getcommgrid());
    FullyDistVec<int, ElementType> ci(civ, G.getcommgrid());
    FullyDistVec<int, ElementType> vi(viv, G.getcommgrid());

    PSpMat::MPI_DCCols r_50(nrow, ncol, ri, ci, vi);
    r_50(nonisov, nonisov, true);

    std::vector<int> riv1(1, 43);
    std::vector<int> civ1(1, 43);
    std::vector<int> viv1(1, 1);

    FullyDistVec<int, ElementType> ri1(riv1, G.getcommgrid());
    FullyDistVec<int, ElementType> ci1(civ1, G.getcommgrid());
    FullyDistVec<int, ElementType> vi1(viv1, G.getcommgrid());

    PSpMat::MPI_DCCols l_13(nrow, ncol, ri1, ci1, vi1);
    l_13(nonisov, nonisov, true);

    std::vector<int> riv2(1, 79);
    std::vector<int> civ2(1, 79);
    std::vector<int> viv2(1, 1);

    FullyDistVec<int, ElementType> ri2(riv2, G.getcommgrid());
    FullyDistVec<int, ElementType> ci2(civ2, G.getcommgrid());
    FullyDistVec<int, ElementType> vi2(viv2, G.getcommgrid());

    PSpMat::MPI_DCCols l_24(nrow, ncol, ri2, ci2, vi2);
    l_24(nonisov, nonisov, true);

    double t_cons2 = MPI_Wtime();
    if (myrank == 0) {
        cout << "    construct single element matrix takes : " << (t_cons2 - t_cons1) << "\n" << endl;
    }

    // query execution
    {
        // start count time
        double total_computing_1 = MPI_Wtime();

        // ==> step 1
        PSpMat::MPI_DCCols m_50(MPI_COMM_WORLD);
        multPrune<RDFINTINT>(G, r_50, m_50, false, true);

        // ==> step 2
        auto dm_50 = diagonalize(m_50);
        mmul_scalar(dm_50, 13);
        PSpMat::MPI_DCCols m_35(MPI_COMM_WORLD);
        multPrune<RDFINTINT>(G, dm_50, m_35, false, true);

        // ==> step 3
        auto dm_35 = diagonalize(m_35);
        mmul_scalar(dm_35, 6);
        PSpMat::MPI_DCCols m_13(MPI_COMM_WORLD);
        multPrune<RDFINTINT>(tG, dm_35, m_13, false, true);

        // ==> step 4
        multPrune<PTINTINT>(l_13, m_13, m_13, true, false);

        // ==> step 5
        auto dm_13 = diagonalize(m_13, true);
        mmul_scalar(dm_13, 8);
        PSpMat::MPI_DCCols m_43(MPI_COMM_WORLD);
        multPrune<RDFINTINT>(tG, dm_13, m_43, false, true);

        // ==> step 6
        auto dm_43 = diagonalize(m_43);
        mmul_scalar(dm_43, 6);
        PSpMat::MPI_DCCols m_24(MPI_COMM_WORLD);
        multPrune<RDFINTINT>(tG, dm_43, m_24, false, true);

        // ==> step 7
        multPrune<PTINTINT>(l_24, m_24, m_24, true, false);

        // ==> step 8
        auto dm_24 = diagonalize(m_24, true);
        mmul_scalar(dm_24, 4);
        PSpMat::MPI_DCCols m_64(MPI_COMM_WORLD);
        multPrune<RDFINTINT>(G, dm_24, m_64, false, true);

        // ==> step 9
        auto dm_35_1 = diagonalize(m_35, true);
        multPrune<PTINTINT>(dm_35_1, m_64, m_64, true, false);

        // ==> step 10
        auto dm_64 = diagonalize(m_64);
        multPrune<PTINTINT>(m_35, dm_64, m_35, false, true);

        // ==> step 11
        auto dm_64_1 = diagonalize(m_64, true);
        multPrune<PTINTINT>(dm_64_1, m_43, m_43, true, false);

        // ==> step 12
        auto dm_43_1 = diagonalize(m_43, true);
        multPrune<PTINTINT>(dm_43_1, m_35, m_35, true, false);

        // ==> step 13
        auto dm_35_2 = diagonalize(m_35, true);
        multPrune<PTINTINT>(dm_35_2, m_50, m_50, true, false);

        // end count time
        double total_computing_2 = MPI_Wtime();

        printReducedInfo(m_50);

        if (myrank == 0) {
            cout << "total mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
            cout << "total transpose time : " << total_transpose_time << " s" << endl;
            cout << "total prune time : " << total_prune_time << " s" << endl;
            cout << "total reduce time : " << total_reduce_time << " s" << endl;
            cout << "total cons_diag time : " << total_construct_diag_time << " s" << endl;
            cout << "total mult time : " << total_mult_time << " s" << endl;
            cout << "query7 totally takes : " << total_computing_2 - total_computing_1 << " s" << endl;
        }
    }
}

template <class IT, class NT, class DER>
void gatherMatrix(SpParMat<IT, NT, DER> &M, std::vector<IT> &ri, std::vector<IT> &ci, std::vector<NT> &vi)
{
    auto commGrid = M.getcommgrid();

    int proccols = commGrid->GetGridCols();
    int procrows = commGrid->GetGridRows();
    IT totalm = M.getnrow();
    IT totaln = M.getncol();
    IT totnnz = M.getnnz();

//    int flinelen = 0;
//    std::ofstream out;
//    if(commGrid->GetRank() == 0)
//    {
//        std::string s;
//        std::stringstream strm;
//        strm << "%%MatrixMarket matrix coordinate real general" << std::endl;
//        strm << totalm << " " << totaln << " " << totnnz << std::endl;
//        s = strm.str();
//        out.open(filename.c_str(),std::ios_base::trunc);
//        flinelen = s.length();
//        out.write(s.c_str(), flinelen);
//        out.close();
//    }

    int colrank = commGrid->GetRankInProcCol();
    int colneighs = commGrid->GetGridRows();
    IT * locnrows = new IT[colneighs];	// number of rows is calculated by a reduction among the processor column
    locnrows[colrank] = (IT) M.getlocalrows();
    MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(),locnrows, 1, MPIType<IT>(), commGrid->GetColWorld());
    IT roffset = std::accumulate(locnrows, locnrows+colrank, 0);
    delete [] locnrows;

    MPI_Datatype datatype;
    MPI_Type_contiguous(sizeof(std::pair<IT,ElementType>), MPI_CHAR, &datatype);
    MPI_Type_commit(&datatype);

    for(int i = 0; i < procrows; i++)	// for all processor row (in order)
    {
        if(commGrid->GetRankInProcCol() == i)	// only the ith processor row
        {
            auto spSeq = M.seqptr();
            IT localrows = spSeq->getnrow();    // same along the processor row
            std::vector< std::vector< std::pair<IT,ElementType> > > csr(localrows);
            if(commGrid->GetRankInProcRow() == 0)	// get the head of processor row
            {
                IT localcols = spSeq->getncol();    // might be different on the last processor on this processor row
                MPI_Bcast(&localcols, 1, MPIType<IT>(), 0, commGrid->GetRowWorld());
                for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over nonempty subcolumns
                {
                    for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
                    {
                        csr[nzit.rowid()].push_back( std::make_pair(colit.colid(), nzit.value()) );
                    }
                }
            }
            else	// get the rest of the processors
            {
                IT n_perproc;
                MPI_Bcast(&n_perproc, 1, MPIType<IT>(), 0, commGrid->GetRowWorld());
                IT noffset = commGrid->GetRankInProcRow() * n_perproc;
                for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over nonempty subcolumns
                {
                    for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
                    {
                        csr[nzit.rowid()].push_back( std::make_pair(colit.colid() + noffset, nzit.value()) );
                    }
                }
            }
            std::pair<IT,ElementType> * ents = NULL;
            int * gsizes = NULL, * dpls = NULL;
            if(commGrid->GetRankInProcRow() == 0)	// only the head of processor row
            {
//                out.open(filename.c_str(),std::ios_base::app);
                gsizes = new int[proccols];
                dpls = new int[proccols]();	// displacements (zero initialized pid)
            }
            for(int j = 0; j < localrows; ++j)
            {
                IT rowcnt = 0;
                sort(csr[j].begin(), csr[j].end());
                int mysize = csr[j].size();
                MPI_Gather(&mysize, 1, MPI_INT, gsizes, 1, MPI_INT, 0, commGrid->GetRowWorld());
                if(commGrid->GetRankInProcRow() == 0)
                {
                    rowcnt = std::accumulate(gsizes, gsizes+proccols, static_cast<IT>(0));
                    std::partial_sum(gsizes, gsizes+proccols-1, dpls+1);
                    ents = new std::pair<IT,ElementType>[rowcnt];	// nonzero entries in the j'th local row
                }

                // int MPI_Gatherv (void* sbuf, int scount, MPI_Datatype stype,
                // 		    void* rbuf, int *rcount, int* displs, MPI_Datatype rtype, int root, MPI_Comm comm)
                MPI_Gatherv(SpHelper::p2a(csr[j]), mysize, datatype, ents, gsizes, dpls, datatype, 0, commGrid->GetRowWorld());
                if(commGrid->GetRankInProcRow() == 0)
                {
                    for(int k=0; k< rowcnt; ++k)
                    {
                        //out << j + roffset + 1 << "\t" << ents[k].first + 1 <<"\t" << ents[k].second << endl;
//                        if (!transpose)
//                            // regular
//                            out << j + roffset + 1 << "\t" << ents[k].first + 1 << "\t";
//                        else
//                            // transpose row/column
//                            out << ents[k].first + 1 << "\t" << j + roffset + 1 << "\t";
//                        out << std::endl;
                        vi.push_back(ents[k].second);
                        ri.push_back(j + roffset);
                        ci.push_back(ents[k].first);
                    }
                    delete [] ents;
                }
            }
            if(commGrid->GetRankInProcRow() == 0)
            {
                DeleteAll(gsizes, dpls);
//                out.close();
            }
        } // end_if the ith processor row
        MPI_Barrier(commGrid->GetWorld());		// signal the end of ith processor row iteration (so that all processors block)
    }
}

int main(int argc, char *argv[]) {
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 1) {
        if (myrank == 0) {
            cout << "Usage: ./lubm10240_l7" << endl;
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
        PSpMat::MPI_DCCols G(MPI_COMM_WORLD);

        double t1 = MPI_Wtime();
        G.ParallelReadMM(Mname, true, selectSecond);
//        G.ReadDistribute(Mname, 0);
        double t2 = MPI_Wtime();

        G.PrintInfo();

        rvec = new FullyDistVec<int, int>(fullWorld);
        rvec->iota(G.getnrow(), 0);
        qvec = new FullyDistVec<int, int>(fullWorld);
        qvec->iota(G.getnrow(), 0);

        float imG = G.LoadImbalance();
        if (myrank == 0) {
            cout << "read file takes : " << (t2 - t1) << " s" << endl;
            cout << "original imbalance : " << imG << endl;
        }

//        std::vector<IndexType> ri, ci;
//        std::vector<ElementType> vi;
//
//        double t3 = MPI_Wtime();
//        gatherMatrix<IndexType, ElementType, PSpMat<ElementType>::DCCols >(G, ri, ci, vi);
//        double t4 = MPI_Wtime();
//        if (myrank == 0) {
//            cout << "gatherMatrix takes : " << (t4 - t3) << endl;
//        }

        lubm10240_L7(G);
    }

    MPI_Finalize();
    return 0;
}
