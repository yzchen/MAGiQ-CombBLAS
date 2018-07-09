#include <iostream>
#include <functional>
#include <algorithm>
#include <sstream>
#include "../include/CombBLAS.h"

#define IndexType uint32_t
#define ElementType int

using namespace std;
using namespace combblas;

class PSpMat {
public:
    typedef SpDCCols<IndexType, ElementType> DCCols;
    typedef SpParMat<IndexType, ElementType, DCCols> MPI_DCCols;
};

static double total_reduce_time = 0.0;
static double total_prune_time = 0.0;
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

// M should have same rows and cols
// indices size should be even, I and J are together
void get_local_indices(PSpMat::MPI_DCCols &M, vector<IndexType> &indices) {
    assert(M.getnrow() == M.getncol());

    auto commGrid = M.getcommgrid();
    int colrank = commGrid->GetRankInProcCol();
    int rowrank = commGrid->GetRankInProcRow();

    int nproc, myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    int colneighs = commGrid->GetGridRows();
    IndexType *locnrows = new IndexType[colneighs];  // number of rows is calculated by a reduction among the processor column
    locnrows[colrank] = M.getlocalrows();
    MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IndexType>(), locnrows, 1, MPIType<IndexType>(), commGrid->GetColWorld());
    IndexType roffset = std::accumulate(locnrows, locnrows + colrank, 0);
    delete[] locnrows;

    int rowneighs = commGrid->GetGridCols();
    IndexType *locncols = new IndexType[rowneighs];  // number of rows is calculated by a reduction among the processor column
    locncols[rowrank] = M.getlocalcols();
    MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IndexType>(), locncols, 1, MPIType<IndexType>(), commGrid->GetRowWorld());
    IndexType coffset = std::accumulate(locncols, locncols + rowrank, 0);
    delete[] locncols;

    //// if there is nothing in current process, then d0 will be NULL pointer
    auto d0 = M.seq().GetInternal();

//    cout << "offset of process " << myrank << ", roffset = " << roffset << ", coffset = " << coffset << endl;

    if (d0 != NULL) {
//        double t1 = MPI_Wtime();
//        I.assign(d0->ir, d0->ir + d0->nz);
//        transform(I.begin(), I.end(), I.begin(), bind2nd(std::plus<int>(), roffset));
//        double t2 = MPI_Wtime();
//        if (myrank == 0) {
//            cout << myrank << ", construct I takes : " << (t2 - t1) << " s" << endl;
//        }

//        double t5 = MPI_Wtime();
        int rind = 0;
        for (int cind = 0; cind < d0->nzc; ++cind) {
            int times = d0->cp[cind + 1] - d0->cp[cind];

            for (int i = 0; i < times; ++i) {
                indices.push_back(d0->ir[rind] + coffset);
                indices.push_back(d0->jc[cind] + roffset);
                rind++;
//                J.push_back(d0->jc[cind]);
            }
        }
//        transform(J.begin(), J.end(), J.begin(), bind2nd(std::plus<int>(), coffset));
//        double t6 = MPI_Wtime();
//        if (myrank == 0) {
//            cout << myrank << ", construct J takes : " << (t6 - t5) << " s" << endl;
//        }

        cout << myrank << "   " << rind << "   " << indices.size() << endl;
        // if does not have same size, wrong
        assert(I.size() == J.size());

    }
//    cout << myrank << " nz : " << I.size() << endl;

}

void send_local_index(shared_ptr<CommGrid> commGrid, vector<IndexType> &Indices){
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    vector<vector<IndexType> > recs;

    int number_count;
    int max_count = 20;
    // large max_count will generate error : bad_alloc
//    int max_count = A.getlocalcols() * A.getlocalrows();

    int colrank = commGrid->GetRankInProcRow();
    int grid_cols = commGrid->GetGridCols();

    if (colrank != 0) {
        cout << myrank << " sender" << endl;
        MPI_Send(&Indices, Indices.size(), MPIType<IndexType>(), 0, 0, commGrid->GetRowWorld());
    } else {
        cout << myrank << " receiver" << endl;
        // first vector will be its self
        recs.push_back(Indices);

        for (int sender = 1; sender < grid_cols; sender++) {
            vector<IndexType> recv_indices(max_count);
            MPI_Status status;

            MPI_Recv(recv_indices.data(), max_count, MPIType<IndexType>(), sender, 0,
                     commGrid->GetRowWorld(), &status);
            MPI_Get_count(&status, MPIType<IndexType>(), &number_count);

//            cout << myrank << " number of numbers : " << number_count << endl;
//            recv_indices.resize(number_count);
            recs.push_back(vector<IndexType>(recv_indices.begin(), recv_indices.begin() + number_count));
        }

        cout << myrank << "   " << recs.size() << endl;
    }
}

void send_local_indices(PSpMat::MPI_DCCols &A, vector<IndexType> &I, vector<IndexType> &J) {
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    auto commGrid = A.getcommgrid();

    int number_count;
    int max_count = 20;
    // large max_count will generate error : bad_alloc
//    int max_count = A.getlocalcols() * A.getlocalrows();

//    cout << myrank << ", " << A.getlocalrows() << " " << A.getlocalcols() << " " << max_count << endl;
    std::vector<IndexType> recv_I(max_count);
    std::vector<IndexType> recv_J(max_count);
//    cout << myrank << " finished data preparation with data " << I.size() << endl;

    int rowneighs = commGrid->GetGridCols();
    int rowrank = commGrid->GetRankInProcRow();

    // prepare data

    for (int p = 2; p <= rowneighs; p *= 2) {

        if (rowrank % p == p / 2) { // this processor is a sender in this round
            number_count = I.size();
//            cout << myrank << ", size of nz : " << I.size() << endl;

            int receiver = rowrank - ceil(p / 2);
            MPI_Send(I.data(), number_count, MPIType<IndexType>(), receiver, 0,
                     commGrid->GetRowWorld());
            MPI_Send(J.data(), number_count, MPIType<IndexType>(), receiver, 1,
                     commGrid->GetRowWorld());
            //break;
//                cout << "round " << p / 2 << ", " << myrank << " sender" << endl;
        } else if (rowrank % p == 0) { // this processor is a receiver in this round
            MPI_Status status;

            int sender = rowrank + ceil(p / 2);
            if (sender < rowneighs) {
                MPI_Recv(recv_I.data(), max_count, MPIType<IndexType>(), sender, 0,
                         commGrid->GetRowWorld(), &status);
                MPI_Recv(recv_J.data(), max_count, MPIType<ElementType>(), sender, 1,
                         commGrid->GetRowWorld(), MPI_STATUS_IGNORE);

                // do something
                MPI_Get_count(&status, MPI_INT, &number_count);
//                    cout << "round " << p / 2 << ", " << myrank << " receiver " << number_count << endl;

                I.insert(I.end(), recv_I.begin(), recv_I.begin() + number_count);
                J.insert(J.end(), recv_J.begin(), recv_J.begin() + number_count);

//                cout << "round " << p / 2 << " rank " << myrank << " has size of I " << I.size() << " and size of J  "
//                     << J.size() << endl;
            }
        }
    }

}

void send_local_results(shared_ptr<CommGrid> commGrid, int res_size) {
    int rowrank = commGrid->GetRankInProcCol();
    int grid_rows = commGrid->GetGridRows();

    if (rowrank != 0) {     // not myrank 0
        MPI_Send(&res_size, 1, MPIType<int>(), 0, 0, commGrid->GetColWorld());
    } else {    // myrank 0
        int recv_size;
        for (int i = 1; i < grid_rows; i++){
            MPI_Recv(&recv_size, 1, MPIType<int>(), i, 0,
                     commGrid->GetColWorld(), MPI_STATUS_IGNORE);
            res_size += recv_size;

            cout << "final size : " << res_size << endl;
        }
    }
}

//void join_l5(shared_ptr<CommGrid> commGrid, vector<vector<IndexType> > &Indices_20, vector<vector<IndexType> > &Indices_21){
//    int myrank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
//
//    int colrank = commGrid->GetRankInProcRow();
//
//    if (colrank == 0) {
//        cout << myrank << " " << Indices_20.size() << " " << Indices_21.size() << endl;
//
////        stringstream os;
////        os << "l5resgen/" << myrank << ".txt";
////
////        double t7 = MPI_Wtime();
////        std::ofstream outFile(os.str());
////        for (int i = 0; i < I_20.size(); i++)
////            outFile << I_20[i] << "\t" << J_20[i] << "\t" << I_21[i] << "\t" << J_21[i] << "\n";
////        double t8 = MPI_Wtime();
////        cout << "output indices results for process " << myrank << " takes : " << (t8 - t7) << " s" << endl;
//
//        vector<vector<int> > res;
//        for (int ind_20 = 0, ind_21 = 0; ind_20 < I_20.size() && ind_21 < I_21.size(); ind_20++) {
//            while( I_21[ind_21] < I_20[ind_20] ){
//                ind_21++;
//            }
//
//            if( I_21[ind_21] == I_20[ind_20] ){
//                res.push_back(vector<int>{J_20[ind_20], J_21[ind_21], I_20[ind_20]});
//            }
//        }
//
//        cout << myrank << " size of res : " << res.size() << endl;
//
//        send_local_results(commGrid, res.size());
//    }
//}

void resGen(PSpMat::MPI_DCCols &m_20, PSpMat::MPI_DCCols &m_12) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (myrank == 0) {
        cout << "---------------------------------------------------------------" << endl;
        cout << "begin result generation ......" << endl;
    }

    // m_12 becoms m_21
//    m_12.Transpose();

    vector<IndexType> Indices_20;
    get_local_indices(m_20, Indices_20);
    send_local_index(m_20.getcommgrid(), Indices_20);

//    vector<IndexType> Indices_21;
//    get_local_indices(m_12, Indices_21);
//    send_local_index(m_12.getcommgrid(), Indices_21);

    // real distributed join phase
//    join_l5(m_20.getcommgrid(), Indices_20, Indices_21);
}

void lubm10240_l5(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG, FullyDistVec<IndexType, IndexType> &nonisov) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    total_reduce_time = 0.0;
    total_prune_time = 0.0;
//    total_construct_diag_time = 0.0;
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

    resGen(m_20, m_12);
}


int main(int argc, char *argv[]) {
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 1) {
        if (myrank == 0) {
            cout << "Usage: ./l5resgen" << endl;
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

        FullyDistVec<IndexType, IndexType> nonisov(commWorld);
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

        lubm10240_l5(G, tG, nonisov);


    }

    MPI_Finalize();
    return 0;
}
