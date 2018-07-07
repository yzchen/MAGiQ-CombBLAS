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

void write_local_vector(vector<IndexType> &recs, string name, int step) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    stringstream os;
    os << "l6resgen/" << name << "/" << myrank << "_3.txt";

//    double t7 = MPI_Wtime();
    std::ofstream outFile(os.str());
    for (int i = 0; i < recs.size(); i += step) {
        for (int ii = 0; ii < step; ii++)
            outFile << recs[i + ii] + 1 << "\t";
        outFile << "\n";
    }

//    double t8 = MPI_Wtime();
//    cout << "output indices results for process " << myrank << " takes : " << (t8 - t7) << " s" << endl;

}

// M should have same rows and cols
// indices size should be even, I and J are together
void get_local_inices(PSpMat::MPI_DCCols &M, vector<IndexType> &indices) {
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
                indices.push_back(d0->ir[rind] + roffset);
                indices.push_back(d0->jc[cind] + coffset);
                rind++;
//                J.push_back(d0->jc[cind]);
            }
        }
//        transform(J.begin(), J.end(), J.begin(), bind2nd(std::plus<int>(), coffset));
//        double t6 = MPI_Wtime();
//        if (myrank == 0) {
//            cout << myrank << ", construct J takes : " << (t6 - t5) << " s" << endl;
//        }

//        cout << myrank << "   " << rind << "   " << indices.size() << endl;
        // if does not have same size, wrong
        assert(I.size() == J.size());

//        stringstream os;
//        os << "l6resgen/" << myrank << ".txt";
//
//        double t7 = MPI_Wtime();
//        std::ofstream outFile(os.str());
//        for (int i = 0; i < indices.size(); i += 2) {
//            outFile << indices[i] + 1 << "\t" << indices[i + 1] + 1 << "\n";
//        }
//        double t8 = MPI_Wtime();
//        cout << "output indices results for process " << myrank << " takes : " << (t8 - t7) << " s" << endl;

    }
//    cout << myrank << " nz : " << I.size() << endl;

}

void merge_local_vectors(vector<IndexType> &first, vector<IndexType> &second, int pair_size1, int pair_size2, int key1,
                         int key2) {
    int ssz1 = first.size(), ssz2 = second.size();
    int i = 0, j = 0;

    vector<IndexType> res;
    res.reserve(ssz1 + ssz2);
//    cout << ssz1 << " " << ssz2 << endl;

    auto it1 = first.begin(), it2 = second.begin();
    while (i < ssz1 && j < ssz2) {
        if (first[i + key1] <= second[j + key2]) {
            res.insert(res.end(), it1 + i, it1 + i + pair_size1);
            i += pair_size1;
        } else {
            res.insert(res.end(), it2 + j, it2 + j + pair_size2);
            j += pair_size2;
        }

//        cout << "size of res : " << res.size() << endl;
    }

    while (i < ssz1) {
        res.insert(res.end(), it1 + i, it1 + i + pair_size1);
        i += pair_size1;
    }

    while (j < ssz2) {
        res.insert(res.end(), it2 + j, it2 + j + pair_size2);
        j += pair_size2;
    }

    first.assign(res.begin(), res.end());
}


void send_local_indices(shared_ptr<CommGrid> commGrid, vector<IndexType> &local_indices) {
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int number_count;
    int max_count = 200000;
    // large max_count will generate error : bad_alloc
//    int max_count = A.getlocalcols() * A.getlocalrows();

//    cout << myrank << ", " << A.getlocalrows() << " " << A.getlocalcols() << " " << max_count << endl;
//    cout << myrank << " finished data preparation with data " << I.size() << endl;

    int colneighs = commGrid->GetGridRows();
    int colrank = commGrid->GetRankInProcCol();

    // prepare data

    for (int p = 2; p <= colneighs; p *= 2) {

        if (colrank % p == p / 2) { // this processor is a sender in this round
            number_count = local_indices.size();
//            cout << myrank << ", size of nz : " << I.size() << endl;

            int receiver = colrank - ceil(p / 2);
            MPI_Send(local_indices.data(), number_count, MPIType<IndexType>(), receiver, 0,
                     commGrid->GetColWorld());
//            cout << "round " << p / 2 << ", " << myrank << " sender " << number_count << endl;
        } else if (colrank % p == 0) { // this processor is a receiver in this round
            MPI_Status status;
            std::vector<IndexType> recv_I(max_count);

            int sender = colrank + ceil(p / 2);
            if (sender < colneighs) {
                MPI_Recv(recv_I.data(), max_count, MPIType<IndexType>(), sender, 0,
                         commGrid->GetColWorld(), &status);

                // do something
                MPI_Get_count(&status, MPI_INT, &number_count);
//                cout << "round " << p / 2 << ", " << myrank << " receiver " << number_count << endl;

                recv_I.resize(number_count);
                merge_local_vectors(local_indices, recv_I, 2, 2, 1, 1);
//                cout << "round " << p / 2 << " rank " << myrank << " has size of indices " << local_indices.size()
//                     << endl;
            }
        }
    }

}

void local_join(shared_ptr<CommGrid> commGrid, vector<IndexType> &indices1, vector<IndexType> &indices2, int pair_size1,
                int pair_size2, int key1,
                int key2, vector<IndexType> &order, vector<IndexType> &res) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int rowrank = commGrid->GetRankInProcCol();

    if (rowrank == 0) {
//        cout << myrank << " has size " << indices1.size() << " and " << indices2.size() << " pair size : " << pair_size1
//             << endl;
        for (int i1 = 0, i2 = 0; i1 < indices1.size() && i2 < indices2.size(); i1 += pair_size1) {
            while (indices1[i1 + key1] > indices2[i2 + key2]) {
                i2 += pair_size2;
            }

            if (indices1[i1 + key1] == indices2[i2 + key2]) {
                for (int oi = 0; oi < order.size(); oi += 2) {
                    // order[oi] == 0 or 1
                    if (order[oi] == 0) {
                        res.push_back(indices1[i1 + order[oi + 1]]);
                    } else {
                        res.push_back(indices2[i2 + order[oi + 1]]);
                    }
                }
//                cout << myrank << " size of res join : " << res.size() << " i1, i2 : " << i1 << ", " << i2 << endl;
            }
        }
    }
}

void local_redistribution(shared_ptr<CommGrid> commGrid, PSpMat::MPI_DCCols &M, vector<IndexType> &range_table,
                          int pivot_range, int pivot,
                          vector<IndexType> &res) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int colrank = commGrid->GetRankInProcRow();
    int rowrank = commGrid->GetRankInProcCol();

    // get range
    if (colrank == 0) {
        // get offset
        int rowneighs = commGrid->GetGridCols();
        IndexType coffset[rowneighs];
        IndexType *locncols = new IndexType[rowneighs];  // number of rows is calculated by a reduction among the processor column
        locncols[rowrank] = M.getlocalcols();
        MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IndexType>(), locncols, 1, MPIType<IndexType>(),
                      commGrid->GetRowWorld());
        coffset[rowrank] = std::accumulate(locncols, locncols + rowrank, 0);
        delete[] locncols;

        MPI_Bcast(&coffset, 1, MPIType<IndexType>(), myrank, MPI_COMM_WORLD);
    }

}


void resGen(PSpMat::MPI_DCCols &m_30, PSpMat::MPI_DCCols &m_43, PSpMat::MPI_DCCols &m_14, PSpMat::MPI_DCCols &m_24) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (myrank == 0) {
        cout << "---------------------------------------------------------------" << endl;
        cout << "begin result generation ......" << endl;
    }

    // m_30 becoms m_03
    m_30.Transpose();
    vector<IndexType> index_03;
    get_local_inices(m_30, index_03);
    send_local_indices(m_30.getcommgrid(), index_03);
    write_local_vector(index_03, "m_30", 2);

//    m_43.Transpose();
    vector<IndexType> index_43;
    get_local_inices(m_43, index_43);
    send_local_indices(m_43.getcommgrid(), index_43);
    write_local_vector(index_43, "m_43", 2);

    vector<IndexType> order1 = {0, 0, 0, 1, 2, 0};
    vector<IndexType> res1;
    local_join(m_30.getcommgrid(), index_03, index_43, 2, 2, 1, 1, order1, res1);
    write_local_vector(res1, "res1", 3);

//
//    vector<IndexType> index_14;
//    vector<IndexType> indices_14;
//    get_local_inices(m_14, index_14);
//    send_local_indices(m_14.getcommgrid(), index_14);
//    write_local_vector(index_14, "m_14");
//
//    vector<IndexType> index_24;
//    vector<IndexType> indices_24;
//    get_local_inices(m_24, index_24);
//    send_local_indices(m_24.getcommgrid(), index_24);
//    write_local_vector(index_24, "m_24");

    // real distributed join phase
//    join_l5(m_20.getcommgrid(), Indices_20, Indices_21);
}

void lubm10240_l6(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG, FullyDistVec<IndexType, IndexType> &nonisov) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    total_reduce_time = 0.0;
    total_prune_time = 0.0;
    total_mmul_scalar_time = 0.0;
    total_dim_apply_time = 0.0;

    auto commWorld = G.getcommgrid();

    FullyDistVec<IndexType, ElementType> dm_30(commWorld), dm_43(commWorld), dm_14(commWorld), dm_24(commWorld);

    auto m_30(G), m_43(tG), m_14(tG), m_24(tG);

    IndexType ind1 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(1345)))[0];
    IndexType ind2 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(22638)))[0];
    IndexType ind3 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(40169)))[0];

    FullyDistVec<IndexType, ElementType> r_30(commWorld, G.getnrow(), 0), l_14(commWorld, G.getnrow(), 0), l_24(
            commWorld, G.getnrow(), 0);
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

//    m_30.SaveGathered("l6resgen/m_30/m_30.txt");
//    m_43.SaveGathered("l6resgen/m_43/m_43.txt");
//    m_14.SaveGathered("l6resgen/m_14/m_14.txt");
//    m_24.SaveGathered("l6resgen/m_24/m_24.txt");


    resGen(m_30, m_43, m_14, m_24);
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

        lubm10240_l6(G, tG, nonisov);


    }

    MPI_Finalize();
    return 0;
}
