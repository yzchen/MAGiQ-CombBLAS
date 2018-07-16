//
// Created by cheny0l on 10/07/18.
//

#ifndef COMBINATORIAL_BLAS_HEADER10240_H
#define COMBINATORIAL_BLAS_HEADER10240_H

#include <iostream>
#include <functional>
#include <algorithm>
#include <sstream>
#include "CombBLAS.h"

using namespace std;
using namespace combblas;

#define IndexType uint32_t
#define ElementType int

class PSpMat {
public:
    typedef SpDCCols<IndexType, ElementType> DCCols;
    typedef SpParMat<IndexType, ElementType, DCCols> MPI_DCCols;
};

double total_reduce_time = 0.0;
double total_prune_time = 0.0;
double total_mmul_scalar_time = 0.0;
double total_dim_apply_time = 0.0;
double total_get_local_indices_time = 0.0;
double total_send_local_indices_time = 0.0;
double total_local_join_time = 0.0;
double total_local_filter_time = 0.0;
double total_redistribution_time = 0.0;
double total_send_result_time = 0.0;

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
    os << "test/" << name << "/" << myrank << "_3.txt";

    std::ofstream outFile(os.str());
    for (int i = 0; i < recs.size(); i += step) {
        for (int ii = 0; ii < step; ii++)
            outFile << recs[i + ii] + 1 << "\t";
        outFile << "\n";
    }
}

// M should have same rows and cols
// indices size should be even, I and J are together
void get_local_indices(PSpMat::MPI_DCCols &M, vector<IndexType> &indices) {
    double t1 = MPI_Wtime();
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
        int rind = 0;
        for (int cind = 0; cind < d0->nzc; ++cind) {
            int times = d0->cp[cind + 1] - d0->cp[cind];

            for (int i = 0; i < times; ++i) {
                indices.push_back(d0->ir[rind] + roffset);
                indices.push_back(d0->jc[cind] + coffset);
                rind++;
            }
        }
        //        cout << myrank << "   " << rind << "   " << indices.size() << endl;
        // if does not have same size, wrong
        assert(I.size() == J.size());
    }
//    cout << myrank << " nz : " << I.size() << endl;

    double t2 = MPI_Wtime();
    if (myrank == 0) {
        cout << "\tget local indices takes : " << (t2 - t1) << " s" << endl;
    }
    total_get_local_indices_time += (t2 - t1);
}

// r1, r2 are not reachable
void
merge_local_vectors(vector<IndexType> &first, vector<IndexType> &second, int l1, int l2, int r1, int r2, int pair_size1,
                    int pair_size2, int key1, int key2) {
//    cout << " l1 = " << l1 << ", r1 = " << r1 << ",         l2 = " << l2 << ", r2 = " << r2 << endl;
//    int ssz1 = first.size(), ssz2 = second.size();
    int i = l1, j = l2;

    vector<IndexType> res;
    res.reserve((r1 - l1) + (r2 - l2));
//    cout << ssz1 << " " << ssz2 << endl;

    auto it1 = first.begin(), it2 = second.begin();
    while (i < r1 && j < r2) {
        if (first[i + key1] <= second[j + key2]) {
            res.insert(res.end(), it1 + i, it1 + i + pair_size1);
            i += pair_size1;
        } else {
            res.insert(res.end(), it2 + j, it2 + j + pair_size2);
            j += pair_size2;
        }
//        cout << "size of res : " << res.size() << endl;
    }

//    cout << " j = " << j << endl;
    while (i < r1) {
//        cout << "in r1 loop" << endl;
        res.insert(res.end(), it1 + i, it1 + i + pair_size1);
        i += pair_size1;
    }

    while (j < r2) {
//        cout << "in r2 loop" << endl;
        res.insert(res.end(), it2 + j, it2 + j + pair_size2);
        j += pair_size2;
    }

    // first.assign(res.begin(), res.end());
    copy_n(res.begin(), res.size(), first.begin() + l1);
}


void send_local_indices(shared_ptr<CommGrid> commGrid, vector<IndexType> &local_indices) {
    double t1 = MPI_Wtime();

    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int colneighs = commGrid->GetGridRows();
    int colrank = commGrid->GetRankInProcCol();

    // prepare data
    int number_count;
    for (int p = 2; p <= colneighs; p *= 2) {

        if (colrank % p == p / 2) { // this processor is a sender in this round
            number_count = local_indices.size();
//            cout << myrank << ", size of nz : " << I.size() << endl;

            int receiver = colrank - ceil(p / 2);
            MPI_Send(&number_count, 1, MPIType<int>(), receiver, 0,
                     commGrid->GetColWorld());
            MPI_Send(local_indices.data(), number_count, MPIType<IndexType>(), receiver, 0,
                     commGrid->GetColWorld());
//            cout << "round " << p / 2 << ", " << myrank << " sender " << number_count << endl;
        } else if (colrank % p == 0) { // this processor is a receiver in this round
            MPI_Status status;
            std::vector<IndexType> recv_I;

            int sender = colrank + ceil(p / 2);
            if (sender < colneighs) {
                MPI_Recv(&number_count, 1, MPIType<int>(), sender, 0,
                         commGrid->GetColWorld(), MPI_STATUS_IGNORE);

                recv_I.resize(number_count);
                MPI_Recv(recv_I.data(), number_count, MPIType<IndexType>(), sender, 0,
                         commGrid->GetColWorld(), MPI_STATUS_IGNORE);

                // do something
                MPI_Get_count(&status, MPIType<int>(), &number_count);
//                cout << "round " << p / 2 << ", " << myrank << " receiver " << number_count << endl;

                int original_sz = local_indices.size();
                local_indices.resize(original_sz + number_count);
                merge_local_vectors(local_indices, recv_I, 0, 0, original_sz, number_count, 2, 2, 1, 1);
//                cout << "round " << p / 2 << " rank " << myrank << " has size of indices " << local_indices.size()
//                     << endl;
            }
        }
    }
//    cout << "myrank " << myrank << " finish sending" << endl;

    double t2 = MPI_Wtime();
    total_send_local_indices_time += (t2 - t1);
    if (myrank == 0) {
        cout << "\tsend local indices takes : " << (t2 - t1) << " s" << endl;
    }
}

void put_tuple(vector<IndexType> &res, vector<IndexType> &source1, vector<IndexType> &source2, int index1, int index2,
               vector<IndexType> &order) {
    for (int oi = 0; oi < order.size(); oi += 2) {
        // order[oi] == 0 or 1
        if (order[oi] == 0) {
            res.push_back(source1[index1 + order[oi + 1]]);
        } else {
            res.push_back(source2[index2 + order[oi + 1]]);
        }
    }

}

void local_join(shared_ptr<CommGrid> commGrid, vector<IndexType> &indices1, vector<IndexType> &indices2, int pair_size1,
                int pair_size2, int key1, int key2, vector<IndexType> &order, vector<IndexType> &res) {
    double t1 = MPI_Wtime();

    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int colrank = commGrid->GetRankInProcCol();

    if (colrank == 0) {
//        cout << myrank << " has size " << indices1.size() << " and " << indices2.size() << " pair size : " << pair_size1
//             << ", " << pair_size2 << endl;
        int i1 = 0, i2 = 0;
        int sz1 = indices1.size(), sz2 = indices2.size();

        while (i1 < sz1 && i2 < sz2) {
            if (indices1[i1 + key1] < indices2[i2 + key2]) {
                i1 += pair_size1;
            } else if (indices1[i1 + key1] > indices2[i2 + key2]) {
                i2 += pair_size2;
            } else {
                put_tuple(res, indices1, indices2, i1, i2, order);

                int i22 = i2 + pair_size2;
                while (i22 < sz2 && indices1[i1 + key1] == indices2[i22 + key2]) {
                    put_tuple(res, indices1, indices2, i1, i22, order);
                    i22 += pair_size2;
                }

                int i11 = i1 + pair_size1;
                while (i11 < sz1 && indices1[i11 + key1] == indices2[i2 + key2]) {
                    put_tuple(res, indices1, indices2, i11, i2, order);
                    i11 += pair_size1;
                }

                i1 += pair_size1;
                i2 += pair_size2;
            }
        }

    }

    double t2 = MPI_Wtime();
    total_local_join_time += (t2 - t1);
    if (myrank == 0) {
        cout << "\tlocal join takes : " << (t2 - t1) << " s\n" << endl;
    }
}

// key11 and key21 are main keys
void
local_filter(shared_ptr<CommGrid> commGrid, vector<IndexType> &indices1, vector<IndexType> &indices2, int pair_size1,
             int pair_size2, int key11, int key12, int key21, int key22, vector<IndexType> &order,
             vector<IndexType> &res) {
    double t1 = MPI_Wtime();

    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int colrank = commGrid->GetRankInProcCol();

    if (colrank == 0) {
//        cout << myrank << " has size " << indices1.size() << " and " << indices2.size() << " pair size : " << pair_size1
//             << ", " << pair_size2 << endl;
        int i1 = 0, i2 = 0;
        int sz1 = indices1.size(), sz2 = indices2.size();

        while (i1 < sz1 && i2 < sz2) {
            if (indices1[i1 + key11] < indices2[i2 + key21]) {
                i1 += pair_size1;
            } else if (indices1[i1 + key11] > indices2[i2 + key21]) {
                i2 += pair_size2;
            } else {
                if (indices1[i1 + key12] == indices2[i2 + key22]) {
                    put_tuple(res, indices1, indices2, i1, -1, order);
                }

                int i22 = i2 + pair_size2;
                while (i22 < sz2 && indices1[i1 + key11] == indices2[i22 + key21]) {
                    if (indices1[i1 + key12] == indices2[i22 + key22]) {
                        put_tuple(res, indices1, indices2, i1, -1, order);
                    }
                    i22 += pair_size2;
                }

                int i11 = i1 + pair_size1;
                while (i11 < sz1 && indices1[i11 + key11] == indices2[i2 + key21]) {
                    if (indices1[i11 + key12] == indices2[i2 + key22]) {
                        put_tuple(res, indices1, indices2, i11, -1, order);
                    }
                    i11 += pair_size1;
                }

                i1 += pair_size1;
                i2 += pair_size2;
            }
        }

//        cout << myrank << " filter size : " << res.size() << endl;

    }
    double t2 = MPI_Wtime();
    total_local_filter_time += (t2 - t1);
    if (myrank == 0) {
        cout << "\tlocal filter takes : " << (t2 - t1) << " s\n" << endl;
    }
}

// merge sort
// left, right are number of tuples, not number of elements
void local_sort_table(vector<IndexType> &range_table, int left, int right, int pair_size, int pivot) {
    if (left < right) {
        int mid = (left + right) / 2;
//        cout << "mid = " << mid << endl;
        local_sort_table(range_table, left, mid, pair_size, pivot);
        local_sort_table(range_table, mid + 1, right, pair_size, pivot);

        merge_local_vectors(range_table, range_table, left * pair_size, (mid + 1) * pair_size, (mid + 1) * pair_size,
                            (right + 1) * pair_size, pair_size, pair_size, pivot, pivot);
    }
}

// column based operation
void local_redistribution(PSpMat::MPI_DCCols &M, vector<IndexType> &range_table, int pair_size,
                          int pivot, vector<IndexType> &res) {
    double t1 = MPI_Wtime();

    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    auto commGrid = M.getcommgrid();

    // get column offset and broadcast them
    int rowneighs = commGrid->GetGridCols();
//    int colneighs = commGrid->GetGridRows();
    IndexType coffset[rowneighs + 1];

    int colrank = commGrid->GetRankInProcCol();
    int rowrank = commGrid->GetRankInProcRow();

    IndexType *locncols = new IndexType[rowneighs];  // number of rows is calculated by a reduction among the processor column
    locncols[rowrank] = M.getlocalcols();
    MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IndexType>(), locncols, 1, MPIType<IndexType>(), commGrid->GetRowWorld());
    coffset[rowrank] = std::accumulate(locncols, locncols + rowrank, 0);
    MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IndexType>(), coffset, 1, MPIType<IndexType>(), commGrid->GetRowWorld());
    delete[] locncols;

    //// magic number for flag
    coffset[rowneighs] = UINT32_MAX;


//    cout << myrank << ", redis, " << range_table.size() << endl;
    local_sort_table(range_table, 0, range_table.size() / pair_size - 1, pair_size, pivot);
//    cout << myrank << ", redis, after sort,  " << range_table.size() << endl;

    vector<int> lens;
    lens.reserve(rowneighs + 1);
    lens.push_back(0);

    // split table into correct range and send to correct process
    int prev = 0;
    for (int i = 1; i <= rowneighs; ++i) {
        int j;
        for (j = prev; j < range_table.size() && range_table[j + pivot] < coffset[i]; j += pair_size) {
        }

        // vector slice : prev -> j - 3
//        cout << myrank << " cut point : " << j << endl;
        int len = j - prev;
        lens.push_back(len);
        prev = j;
    }

    vector<int> partial_sums(lens);
    partial_sum(lens.begin(), lens.end() - 1, partial_sums.begin());

    int *recvcount[rowneighs];
    vector<int> displs(rowneighs);
    displs[0] = 0;

    if (colrank == 0) {
        recvcount[myrank] = new int[rowneighs];

        for (int i = 0; i < rowneighs; ++i) {
            MPI_Gather(lens.data() + i + 1, 1, MPI_INT, recvcount[i], 1, MPI_INT, i, commGrid->GetRowWorld());
        }


        for (int j = 1; j < rowneighs; ++j) {
            displs[j] = displs[j - 1] + recvcount[myrank][j - 1];
        }

        res.resize(displs[rowneighs - 1] + recvcount[myrank][rowneighs - 1]);
//        cout << myrank << " redis, result size = " << displs[rowneighs - 1] + recvcount[myrank][rowneighs - 1] << endl;

        for (int k = 0; k < rowneighs; ++k) {
            MPI_Gatherv(range_table.data() + partial_sums[k], lens[k + 1], MPIType<IndexType>(), res.data(),
                        recvcount[k],
                        displs.data(), MPIType<IndexType>(), k, commGrid->GetRowWorld());

            if (myrank == k) {
//                cout << "\nafter gatherv " << myrank << " res size : " << res.size() / pair_size << endl;

                local_sort_table(res, 0, res.size() / pair_size - 1, pair_size, pivot);
            }
        }
    }

    double t2 = MPI_Wtime();
    total_redistribution_time += (t2 - t1);
    if (myrank == 0) {
        cout << "\tlocal redistribution takes : " << (t2 - t1) << " s\n" << endl;
    }

}

void send_local_results(shared_ptr<CommGrid> commGrid, unsigned res_size) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int rowneighs = commGrid->GetGridRows();

    if (myrank < rowneighs) {

        if (commGrid->GetRank() != 0) {     // not myrank 0
            MPI_Send(&res_size, 1, MPIType<unsigned>(), 0, 0, commGrid->GetRowWorld());
        } else {    // myrank 0
            int recv_size;
            for (int i = 1; i < rowneighs; i++) {
//                cout << "receive from " << i << endl;
                MPI_Recv(&recv_size, 1, MPIType<unsigned>(), i, 0, commGrid->GetRowWorld(), MPI_STATUS_IGNORE);
                res_size += recv_size;
            }
            cout << "final result size : " << res_size << endl;
            cout << "total get local indices time : " << total_get_local_indices_time << " s" << endl;
            cout << "total send local indices time : " << total_send_local_indices_time << " s" << endl;
            cout << "total local join time : " << total_local_join_time << " s" << endl;
            cout << "total local filter time : " << total_local_filter_time << " s" << endl;
//            cout << "total send local result time : " << total_send_result_time << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }
    }
//    }
}

#endif //COMBINATORIAL_BLAS_HEADER10240_H
