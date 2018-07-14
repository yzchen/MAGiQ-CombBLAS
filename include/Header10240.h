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
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // TODO : magic number
    // TODO : future fix : mpi gather to get all sizes and then add them to get actual total size
    int max_count = 15000000;
//    int max_count = local_indices.size();
    // large max_count will generate error : bad_alloc
//    int max_count = A.getlocalcols() * A.getlocalrows();

//    cout << myrank << ", " << A.getlocalrows() << " " << A.getlocalcols() << " " << max_count << endl;
//    cout << myrank << " finished data preparation with data " << I.size() << endl;

    int colneighs = commGrid->GetGridRows();
    int colrank = commGrid->GetRankInProcCol();

    // prepare max_count
//
////    if (colrank != 0) {
////        number_count = local_indices.size();
////
////        cout << "myrank " << myrank << " sender " <<endl;
////
////        MPI_Send(&number_count, 1, MPIType<int>(), 0, 0, commGrid->GetColWorld());
////
////    } else {
////        cout << "myrank " << myrank << " receiver " <<endl;
////        for (int i = 1; i < colneighs; ++i) {
////            int recv_count;
////            MPI_Recv(&recv_count, 1, MPIType<int>(), i, 0, commGrid->GetColWorld(), MPI_STATUS_IGNORE);
////            if (max_count < recv_count) {
////                max_count = recv_count;
////            }
////        }
////    }
//
//    for (int p = 2; p <= colneighs; p *= 2) {
//
//        if (colrank % p == p / 2) { // this processor is a sender in this round
//            int receiver = colrank - ceil(p / 2);
//            MPI_Send(&max_count, 1, MPIType<int>(), receiver, 0, commGrid->GetColWorld());
//        } else if (colrank % p == 0) { // this processor is a receiver in this round
//            int recv_count;
//
//            int sender = colrank + ceil(p / 2);
//            if (sender < colneighs) {
//                MPI_Recv(&recv_count, 1, MPIType<int>(), sender, 0, commGrid->GetColWorld(), MPI_STATUS_IGNORE);
//                if (max_count < recv_count) {
//                    max_count = recv_count;
//                }
//            }
//        }
//    }
//
//    cout << myrank << " local indices max count in send_local_indices : " << max_count << endl;

    // prepare data
    int number_count;
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
                MPI_Get_count(&status, MPIType<int>(), &number_count);
//                cout << "round " << p / 2 << ", " << myrank << " receiver " << number_count << endl;

                recv_I.resize(number_count);
                int original_sz = local_indices.size();
                local_indices.resize(original_sz + number_count);
                merge_local_vectors(local_indices, recv_I, 0, 0, original_sz, number_count, 2, 2, 1, 1);
//                cout << "round " << p / 2 << " rank " << myrank << " has size of indices " << local_indices.size()
//                     << endl;
            }
        }
    }

//    cout << "myrank " << myrank << " finish sending" << endl;

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
//                if (indices1[i1 + key1] == indices2[i2 + key2]) {
//                    for (int oi = 0; oi < order.size(); oi += 2) {
//                        // order[oi] == 0 or 1
//                        if (order[oi] == 0) {
//                            res.push_back(indices1[i1 + order[oi + 1]]);
//                        } else {
//                            res.push_back(indices2[i2 + order[oi + 1]]);
//                        }
//                    }
//                }

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

//        cout << myrank << " join size : " << res.size() << endl;

//        for (int i1 = 0, i2 = 0; i1 < indices1.size() && i2 < indices2.size(); i1 += pair_size1) {
//            while (indices1[i1 + key1] > indices2[i2 + key2]) {
//                i2 += pair_size2;
//            }
//
//            if (indices1[i1 + key1] == indices2[i2 + key2]) {
//                for (int oi = 0; oi < order.size(); oi += 2) {
//                    // order[oi] == 0 or 1
//                    if (order[oi] == 0) {
//                        res.push_back(indices1[i1 + order[oi + 1]]);
//                    } else {
//                        res.push_back(indices2[i2 + order[oi + 1]]);
//                    }
//                }
////                cout << myrank << " size of res join : " << res.size() << " i1, i2 : " << i1 << ", " << i2 << endl;
//            }
//        }
    }
}

// key11 and key21 are main keys
void
local_filter(shared_ptr<CommGrid> commGrid, vector<IndexType> &indices1, vector<IndexType> &indices2, int pair_size1,
             int pair_size2, int key11, int key12, int key21, int key22, vector<IndexType> &order,
             vector<IndexType> &res) {
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
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    auto commGrid = M.getcommgrid();

    // get column offset and broadcast them
    int rowneighs = commGrid->GetGridCols();
//    int colneighs = commGrid->GetGridRows();
    IndexType coffset[rowneighs + 1];

    int colrank = commGrid->GetRankInProcCol();
    int rowrank = commGrid->GetRankInProcRow();

//    IndexType *locnrows = new IndexType[colneighs];  // number of rows is calculated by a reduction among the processor column
//    locnrows[colrank] = M.getlocalrows();
//    MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IndexType>(), locnrows, 1, MPIType<IndexType>(), commGrid->GetColWorld());
//    IndexType ro = std::accumulate(locnrows, locnrows + colrank, 0);
//    delete[] locnrows;

    IndexType *locncols = new IndexType[rowneighs];  // number of rows is calculated by a reduction among the processor column
    locncols[rowrank] = M.getlocalcols();
    MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IndexType>(), locncols, 1, MPIType<IndexType>(), commGrid->GetRowWorld());
    coffset[rowrank] = std::accumulate(locncols, locncols + rowrank, 0);
    MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IndexType>(), coffset, 1, MPIType<IndexType>(), commGrid->GetRowWorld());
    delete[] locncols;

    //// magic number for flag
    coffset[rowneighs] = UINT32_MAX;

//    if (colrank == 0) {
////        cout << myrank << " mycolumn : " << colrank << " base : " << ", " << co << endl;
//
//        cout << "coffset : " << myrank << "\t";
//        for (auto x : coffset) {
//            cout << x << "\t";
//        }
//        cout << endl;
//    }

//    cout << myrank << ", redis, " << range_table.size() << endl;
    local_sort_table(range_table, 0, range_table.size() / pair_size - 1, pair_size, pivot);
//    cout << myrank << ", redis, after sort,  " << range_table.size() << endl;
//    write_local_vector(range_table, "res12", 3);

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

//        vector<int> recvcounts;
//
//        /* Only root has the received data */
//        if (myrank == i - 1)
//            recvcounts.resize(rowneighs);
//
//        if (colrank == 0) {
//            MPI_Barrier(commGrid->GetRowWorld());
//            MPI_Gather(&len, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, i - 1, commGrid->GetRowWorld());
//
//            if (myrank == i - 1) {
//                cout << myrank << "\t";
//                for (int k = 0; k < rowneighs; ++k) {
//                    cout << recvcounts[k] << "\t";
//                }
//                cout << endl;
//            }
//
//            vector<int> displs(recvcounts);
//            partial_sum(displs.begin(), displs.end(), displs.begin());
//
////            MPI_Gatherv(range_table.data() + prev, len, MPIType<IndexType>(), res.data(), recvcounts.data(), displs.data(), MPIType<IndexType>(), i - 1, commGrid->GetRowWorld());
//            MPI_Barrier(commGrid->GetRowWorld());
//        }

        prev = j;
    }

//    if (colrank == 0) {
//        cout << "lens : " << myrank << "\t";
//        for (int k = 0; k < lens.size(); ++k) {
//            cout << lens[k] << "\t";
//        }
//        cout << endl;
//    }

    vector<int> partial_sums(lens);
    partial_sum(lens.begin(), lens.end() - 1, partial_sums.begin());

//    if (colrank == 0) {
//        cout << "partial_sum : " << myrank << "\t";
//        for (int k = 0; k < partial_sums.size(); ++k) {
//            cout << partial_sums[k] << "\t";
//        }
//        cout << endl;
//    }

    int *recvcount[rowneighs];
    vector<int> displs(rowneighs);
    displs[0] = 0;

    if (colrank == 0) {
        recvcount[myrank] = new int[rowneighs];

        for (int i = 0; i < rowneighs; ++i) {
            MPI_Gather(lens.data() + i + 1, 1, MPI_INT, recvcount[i], 1, MPI_INT, i, commGrid->GetRowWorld());
        }

//        cout << "recvcount : " << myrank << "\t";
//        for (int k = 0; k < rowneighs; ++k) {
//            cout << recvcount[myrank][k] << "\t";
//        }
//        cout << endl;

        for (int j = 1; j < rowneighs; ++j) {
            displs[j] = displs[j - 1] + recvcount[myrank][j - 1];
        }

//        cout << "displs : " << myrank << "\t";
//        for (int k = 0; k < displs.size(); ++k) {
//            cout << displs[k] << "\t";
//        }
//        cout << endl;

        res.resize(displs[rowneighs - 1] + recvcount[myrank][rowneighs - 1]);
//        cout << myrank << " redis, result size = " << displs[rowneighs - 1] + recvcount[myrank][rowneighs - 1] << endl;

//        for (int l = 0; l < rowneighs; ++l) {
//            MPI_Gatherv(range_table.data() + partial_sums[l], lens[l], MPIType<IndexType>(), res.data(), recvcount[l],
//                        displs.data(), MPIType<IndexType>(), l, commGrid->GetRowWorld());
//        }

//        cout << "\nrank " << myrank << " has : " << endl;
//        for (int t = 0; t < lens[1]; t += 3) {
//            cout << range_table[t] << " " << range_table[t + 1] << " " << range_table[t + 2] << "\n";
//        }
//        cout << endl;

//        MPI_Gatherv(range_table.data(), lens[1], MPIType<IndexType>(), res.data(), recvcount[0],
//                    displs.data(), MPIType<IndexType>(), 0, commGrid->GetRowWorld());
//
//        MPI_Gatherv(range_table.data() + partial_sums[1], lens[2], MPIType<IndexType>(), res.data(), recvcount[1],
//                    displs.data(), MPIType<IndexType>(), 1, commGrid->GetRowWorld());
//
//        if (myrank <= 1) {
//                cout << "\nafter gatherv" << endl;
//                local_sort_table(res, 0, res.size() / 3 - 1, pair_size, pivot);
//                cout << "rank " << myrank << " has : " << endl;
//                for (int t = 0; t < res.size(); t += 3) {
//                    cout << res[t] << " " << res[t + 1] << " " << res[t + 2] << "\n";
//                }
//                cout << endl;
//
//        }

        for (int k = 0; k < rowneighs; ++k) {
            MPI_Gatherv(range_table.data() + partial_sums[k], lens[k + 1], MPIType<IndexType>(), res.data(),
                        recvcount[k],
                        displs.data(), MPIType<IndexType>(), k, commGrid->GetRowWorld());

            if (myrank == k) {
//                cout << "\nafter gatherv " << myrank << " res size : " << res.size() / pair_size << endl;

                local_sort_table(res, 0, res.size() / pair_size - 1, pair_size, pivot);

//                cout << "\nafter gatherv sorting " << myrank << endl;
//                cout << "rank " << k << " has : " << endl;
//                for (int t = 0; t < res.size(); t += 3) {
//                    cout << res[t] << " " << res[t + 1] << " " << res[t + 2] << "\n";
//                }
//                cout << endl;
            }
        }

//        write_local_vector(res, "res13", 3);
    }

}

void send_local_results(shared_ptr<CommGrid> commGrid, unsigned res_size) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

//    if (commGrid->GetRankInProcCol() == 0) {
//    cout << commGrid->GetRank() << " begin sending results1" << endl;
    int rowneighs = commGrid->GetGridRows();
//    cout << commGrid->GetRank() << " begin sending results2" << endl;

    if (myrank < rowneighs) {

        if (commGrid->GetRank() != 0) {     // not myrank 0
//            cout << commGrid->GetRank() << " before sending " << endl;
            MPI_Send(&res_size, 1, MPIType<unsigned>(), 0, 0, commGrid->GetRowWorld());
//            cout << commGrid->GetRank() << " after sending " << endl;
        } else {    // myrank 0
            int recv_size;
            for (int i = 1; i < rowneighs; i++) {
//                cout << "receive from " << i << endl;
                MPI_Recv(&recv_size, 1, MPIType<unsigned>(), i, 0, commGrid->GetRowWorld(), MPI_STATUS_IGNORE);
                res_size += recv_size;
            }
            cout << "final result size : " << res_size << endl;
            cout << "---------------------------------------------------------------" << endl;
        }
    }
//    }
}


#endif //COMBINATORIAL_BLAS_HEADER10240_H
