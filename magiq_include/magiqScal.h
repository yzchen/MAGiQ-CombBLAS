//
// Created by cheny0l on 24/07/18.
//

#ifndef COMBINATORIAL_BLAS_HEADER_SCAL_H
#define COMBINATORIAL_BLAS_HEADER_SCAL_H

#include <iostream>
#include <functional>
#include <algorithm>
#include <sstream>
#include "../include/CombBLAS.h"
#include "../external/pmergesort/src/pmergesort.h"

using namespace std;
using namespace combblas;

// IndexType is used for index in CombBLAS sparse matrix and dense vectors,
// for dataset which has less than 2^32-1(= 4B) vertices, can change it to uint32_t,
// starting from lubm13B, uint64_t is needed
#define IndexType uint64_t
#define ElementType uint8_t

// Sparse matrix definition
class PSpMat {
public:
    typedef SpDCCols<IndexType, ElementType> DCCols;
    typedef SpParMat<IndexType, ElementType, DCCols> MPI_DCCols;
};

// time for running queries
static double total_reduce_time = 0.0;
static double total_prune_time = 0.0;
static double total_mmul_scalar_time = 0.0;
static double total_dim_apply_time = 0.0;

// time for result generation
static double total_get_local_indices_time = 0.0;
static double total_send_local_indices_time = 0.0;
static double total_local_join_time = 0.0;
static double total_local_filter_time = 0.0;
static double total_redistribution_time = 0.0;

// comparasion struct for qsort in result generation
typedef struct {
    IndexType a, b, c;
} Int3;

int compInt3A(const void *elem1, const void *elem2) {
    Int3 f = *((Int3 *) elem1);
    Int3 s = *((Int3 *) elem2);
    if (f.a > s.a) return 1;
    if (f.a < s.a) return -1;
    return 0;
}

int compInt3B(const void *elem1, const void *elem2) {
    Int3 f = *((Int3 *) elem1);
    Int3 s = *((Int3 *) elem2);
    if (f.b > s.b) return 1;
    if (f.b < s.b) return -1;
    return 0;
}

int compInt3C(const void *elem1, const void *elem2) {
    Int3 f = *((Int3 *) elem1);
    Int3 s = *((Int3 *) elem2);
    if (f.c > s.c) return 1;
    if (f.c < s.c) return -1;
    return 0;
}

typedef struct {
    IndexType a, b, c, d;
} Int4;

int compInt4A(const void *elem1, const void *elem2) {
    Int4 f = *((Int4 *) elem1);
    Int4 s = *((Int4 *) elem2);
    if (f.a > s.a) return 1;
    if (f.a < s.a) return -1;
    return 0;
}

int compInt4B(const void *elem1, const void *elem2) {
    Int4 f = *((Int4 *) elem1);
    Int4 s = *((Int4 *) elem2);
    if (f.b > s.b) return 1;
    if (f.b < s.b) return -1;
    return 0;
}

int compInt4C(const void *elem1, const void *elem2) {
    Int4 f = *((Int4 *) elem1);
    Int4 s = *((Int4 *) elem2);
    if (f.c > s.c) return 1;
    if (f.c < s.c) return -1;
    return 0;
}

int compInt4D(const void *elem1, const void *elem2) {
    Int4 f = *((Int4 *) elem1);
    Int4 s = *((Int4 *) elem2);
    if (f.d > s.d) return 1;
    if (f.d < s.d) return -1;
    return 0;
}

typedef struct {
    IndexType a, b, c, d, e;
} Int5;

int compInt5A(const void *elem1, const void *elem2) {
    Int5 f = *((Int5 *) elem1);
    Int5 s = *((Int5 *) elem2);
    if (f.a > s.a) return 1;
    if (f.a < s.a) return -1;
    return 0;
}

int compInt5B(const void *elem1, const void *elem2) {
    Int5 f = *((Int5 *) elem1);
    Int5 s = *((Int5 *) elem2);
    if (f.b > s.b) return 1;
    if (f.b < s.b) return -1;
    return 0;
}

int compInt5C(const void *elem1, const void *elem2) {
    Int5 f = *((Int5 *) elem1);
    Int5 s = *((Int5 *) elem2);
    if (f.c > s.c) return 1;
    if (f.c < s.c) return -1;
    return 0;
}

int compInt5D(const void *elem1, const void *elem2) {
    Int5 f = *((Int5 *) elem1);
    Int5 s = *((Int5 *) elem2);
    if (f.d > s.d) return 1;
    if (f.d < s.d) return -1;
    return 0;
}

int compInt5E(const void *elem1, const void *elem2) {
    Int5 f = *((Int5 *) elem1);
    Int5 s = *((Int5 *) elem2);
    if (f.e > s.e) return 1;
    if (f.e < s.e) return -1;
    return 0;
}

// comparasion function pointer array
int (*comp[15])(const void *, const void *);

bool isZero(ElementType t) { return t == 0; }

// special semiring for dimApply
ElementType rdf_multiply(ElementType a, ElementType b) {
    if (a != 0 && b != 0 && a == b) {   return static_cast<ElementType>(1);
                            } else {    return static_cast<ElementType>(0);     }
}

// handle duplicate in original loaded data
ElementType selectSecond(ElementType a, ElementType b) { return b; }

PSpMat::MPI_DCCols transpose(const PSpMat::MPI_DCCols &M) {
    PSpMat::MPI_DCCols N(M);
    N.Transpose();
    return N;
}

// reset all time reslated to query execution
void clear_query_time() {
    total_reduce_time = 0.0;
    total_prune_time = 0.0;
    total_mmul_scalar_time = 0.0;
    total_dim_apply_time = 0.0;
}

void permute(PSpMat::MPI_DCCols &G, FullyDistVec<IndexType, IndexType> &nonisov) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // permute G
    double t_perm1 = MPI_Wtime();

    nonisov.iota(G.getnrow(), 0);
    nonisov.RandPerm();

    G(nonisov, nonisov, true);
    double t_perm2 = MPI_Wtime();

    float impG = G.LoadImbalance();
#ifdef MAGIQ_DEBUG
    if (myrank == 0) {
        cout << "\tpermutation takes : " << (t_perm2 - t_perm1) << " s" << endl;
        cout << "\timbalance of permuted G : " << impG << endl;
    }
#endif
}

// diagonalize based on Row/Column, then scale the FullyDistVec with scalar
void diagonalizeV(const PSpMat::MPI_DCCols &M, FullyDistVec<IndexType, ElementType> &diag, Dim dim = Row, ElementType scalar = 1) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t1 = MPI_Wtime();
    M.Reduce(diag, dim, std::logical_or<ElementType>(), static_cast<ElementType>(0));
    double t2 = MPI_Wtime();

    double t3 = MPI_Wtime();
    if (scalar != 1) {      diag.Apply(bind2nd(multiplies<ElementType>(), scalar));     }
    double t4 = MPI_Wtime();

    total_reduce_time += (t2 - t1);
    total_mmul_scalar_time += (t4 - t3);
#ifdef MAGIQ_DEBUG
    if (myrank == 0) {
        cout << "\tdiag-reduce takes : " << (t2 - t1) << " s" << endl;
        cout << "\tmmul-scalar takes : " << (t4 - t3) << " s" << endl;
    }
#endif
}

// dimApply and prune, isRDF indicates the semiring
void multDimApplyPrune(PSpMat::MPI_DCCols &A, FullyDistVec<IndexType, ElementType> &v, Dim dim, bool isRDF) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t1 = MPI_Wtime();
    if (isRDF) {    A.DimApply(dim, v, rdf_multiply);
        } else {    A.DimApply(dim, v, std::multiplies<ElementType>());     }
    double t2 = MPI_Wtime();

    double t3 = MPI_Wtime();
    A.Prune(isZero);
    double t4 = MPI_Wtime();

    total_dim_apply_time += (t2 - t1);
    total_prune_time += (t4 - t3);
#ifdef MAGIQ_DEBUG
    // printInfo after each computation step
    A.PrintInfo();

    if (myrank == 0) {
        cout << "\tdim-apply takes: " << (t2 - t1) << " s" << endl;
        cout << "\tprune takes: " << (t4 - t3) << " s" << endl;
    }
#endif
}

// reset all time related to result generation
void clear_result_time() {
    total_get_local_indices_time = 0.0;
    total_send_local_indices_time = 0.0;
    total_local_join_time = 0.0;
    total_local_filter_time = 0.0;
    total_redistribution_time = 0.0;
}

//////////////////////////////////////////////////////////////////////////////////
//                          Result Generation Phase                             //
//////////////////////////////////////////////////////////////////////////////////

// M should have same rows and cols
// before this function call, indices should be empty
// after this function call, indices size should be even, I and J are together
void get_local_indices(PSpMat::MPI_DCCols &M, vector<IndexType> &indices) {
    double t1 = MPI_Wtime();
    assert(M.getnrow() == M.getncol());

    indices.clear();

    auto commGrid = M.getcommgrid();
    int colrank = commGrid->GetRankInProcCol();
    int rowrank = commGrid->GetRankInProcRow();

    // calculate row offset for each process
    int colneighs = commGrid->GetGridRows();
    IndexType *locnrows = new IndexType[colneighs];  // number of rows is calculated by a reduction among the processor column
    locnrows[colrank] = M.getlocalrows();
    MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IndexType>(), locnrows, 1, MPIType<IndexType>(), commGrid->GetColWorld());
    IndexType roffset = std::accumulate(locnrows, locnrows + colrank, static_cast<IndexType>(0));
    delete[] locnrows;

    // calculate column offset for each process
    int rowneighs = commGrid->GetGridCols();
    IndexType *locncols = new IndexType[rowneighs];  // number of columns is calculated by a reduction among the processor row
    locncols[rowrank] = M.getlocalcols();
    MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IndexType>(), locncols, 1, MPIType<IndexType>(), commGrid->GetRowWorld());
    IndexType coffset = std::accumulate(locncols, locncols + rowrank, static_cast<IndexType>(0));
    delete[] locncols;

    // if there is nothing in current process, then d0 will be NULL pointer
    auto d0 = M.seq().GetInternal();
    if (d0 != NULL) {
        indices.reserve(d0->nz);
        IndexType rind = 0;
        for (IndexType cind = 0; cind < d0->nzc; ++cind) {
            IndexType times = d0->cp[cind + 1] - d0->cp[cind];

            for (IndexType i = 0; i < times; ++i) {
                indices.push_back(d0->ir[rind] + roffset);
                indices.push_back(d0->jc[cind] + coffset);
                rind++;
            }
        }
        // if does not have same size, must be wrong
        // assert(I.size() == J.size() && "When get local indices, I and J should have same size.");
    }

    double t2 = MPI_Wtime();
#ifdef MAGIQ_DEBUG
    if (commGrid->GetRank() == 0) {
        cout << "\tget local indices takes : " << (t2 - t1) << " s" << endl;
    }
#endif
    total_get_local_indices_time += (t2 - t1);
}

// r1, r2 are not reachable
// first should have enough size to contain the sorted data
void merge_local_vectors(vector<IndexType> &first, vector<IndexType> &second, IndexType l1, IndexType l2, IndexType r1, IndexType r2, 
                        int pair_size1, int pair_size2, int key1, int key2) {
    IndexType i = l1, j = l2;

    vector<IndexType> res;
    res.reserve((r1 - l1) + (r2 - l2));

    auto it1 = first.begin(), it2 = second.begin();
    while (i < r1 && j < r2) {
        if (first[i + key1] <= second[j + key2]) {
            res.insert(res.end(), it1 + i, it1 + i + pair_size1);   i += pair_size1;
        } else {
            res.insert(res.end(), it2 + j, it2 + j + pair_size2);   j += pair_size2;
        }
    }

    while (i < r1) {
        res.insert(res.end(), it1 + i, it1 + i + pair_size1);   i += pair_size1;
    }
    while (j < r2) {
        res.insert(res.end(), it2 + j, it2 + j + pair_size2);   j += pair_size2;
    }

    copy_n(res.begin(), res.size(), first.begin() + l1);
}

// each process send its local indices to first process in current column
void send_local_indices(shared_ptr<CommGrid> commGrid, vector<IndexType> &local_indices) {
    double t1 = MPI_Wtime();

    int colneighs = commGrid->GetGridRows();
    int colrank = commGrid->GetRankInProcCol();

    // prepare data
    // number of elements should not exceed int32_max, mpi cannot directly send more than int32_max elements
    // binary tree merging and sending
    int number_count;
    for (int p = 2; p <= colneighs; p *= 2) {
        if (colrank % p == p / 2) {         // this processor is a sender in this round
            number_count = local_indices.size();

            int receiver = colrank - ceil(p / 2);
            MPI_Send(&number_count, 1, MPIType<int>(), receiver, 0, commGrid->GetColWorld());
            MPI_Send(local_indices.data(), number_count, MPIType<IndexType>(), receiver, 0, commGrid->GetColWorld());
        } else if (colrank % p == 0) {      // this processor is a receiver in this round
            std::vector<IndexType> recv_I;

            int sender = colrank + ceil(p / 2);
            if (sender < colneighs) {
                MPI_Recv(&number_count, 1, MPIType<int>(), sender, 0, commGrid->GetColWorld(), MPI_STATUS_IGNORE);
                recv_I.resize(number_count);
                MPI_Recv(recv_I.data(), number_count, MPIType<IndexType>(), sender, 0, commGrid->GetColWorld(), MPI_STATUS_IGNORE);
                IndexType original_sz = local_indices.size();
                local_indices.resize(original_sz + number_count);
                merge_local_vectors(local_indices, recv_I, 0, 0, original_sz, number_count, 2, 2, 1, 1);
            }
        }
    }

    double t2 = MPI_Wtime();
    total_send_local_indices_time += (t2 - t1);
#ifdef MAGIQ_DEBUG
    if (commGrid->GetRank() == 0) {
        cout << "\tsend local indices takes : " << (t2 - t1) << " s" << endl;
    }
#endif
}

void put_tuple(vector<IndexType> &res, vector<IndexType> &source1, vector<IndexType> &source2,
               IndexType index1, IndexType index2, vector<IndexType> &order) {
    for (IndexType oi = 0; oi < order.size(); oi += 2) {
        // order[oi] == 0 or 1, other choices will be treated as 1
        if (order[oi] == 0) {   res.push_back(source1[index1 + order[oi + 1]]);
        } else  {   res.push_back(source2[index2 + order[oi + 1]]); }
    }
}

// local join with special tables, only processors in first row work
void local_join(shared_ptr<CommGrid> commGrid, vector<IndexType> &indices1, vector<IndexType> &indices2, int pair_size1,
                int pair_size2, int key1, int key2, vector<IndexType> &order, vector<IndexType> &res) {
    double t1 = MPI_Wtime();

    res.clear();
    res.shrink_to_fit();

    if (commGrid->GetRankInProcCol() == 0) {        // first row processors
        IndexType i1 = 0, i2 = 0;
        IndexType sz1 = indices1.size(), sz2 = indices2.size();

        while (i1 < sz1 && i2 < sz2) {
            if (indices1[i1 + key1] < indices2[i2 + key2]) {
                i1 += pair_size1;
            } else if (indices1[i1 + key1] > indices2[i2 + key2]) {
                i2 += pair_size2;
            } else {
                put_tuple(res, indices1, indices2, i1, i2, order);

                IndexType i22 = i2 + pair_size2;
                while (i22 < sz2 && indices1[i1 + key1] == indices2[i22 + key2]) {
                    put_tuple(res, indices1, indices2, i1, i22, order);
                    i22 += pair_size2;
                }

                IndexType i11 = i1 + pair_size1;
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
#ifdef MAGIQ_DEBUG
    if (commGrid->GetRank() == 0) {
        cout << "\tlocal join takes : " << (t2 - t1) << " s\n" << endl;
    }
#endif
}

// key11 and key21 should be main keys, tables should be sorted based on them
void local_filter(shared_ptr<CommGrid> commGrid, vector<IndexType> &indices1, vector<IndexType> &indices2, int pair_size1,
                  int pair_size2, int key11, int key12, int key21, int key22, vector<IndexType> &order, vector<IndexType> &res) {
    double t1 = MPI_Wtime();

    res.clear();
    res.reserve(2000);      // 2000 is a magic number, just to somehow reserve memory

    if (commGrid->GetRankInProcCol() == 0) {
        IndexType i1 = 0, i2 = 0;
        IndexType sz1 = indices1.size(), sz2 = indices2.size();

        while (i1 < sz1 && i2 < sz2) {
            if (indices1[i1 + key11] < indices2[i2 + key21]) {
                i1 += pair_size1;
            } else if (indices1[i1 + key11] > indices2[i2 + key21]) {
                i2 += pair_size2;
            } else {
                if (indices1[i1 + key12] == indices2[i2 + key22]) {
                    put_tuple(res, indices1, indices2, i1, -1, order);
                }

                IndexType i22 = i2 + pair_size2;
                while (i22 < sz2 && indices1[i1 + key11] == indices2[i22 + key21]) {
                    if (indices1[i1 + key12] == indices2[i22 + key22]) {
                        put_tuple(res, indices1, indices2, i1, -1, order);
                    }
                    i22 += pair_size2;
                }

                IndexType i11 = i1 + pair_size1;
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
    }

    double t2 = MPI_Wtime();
    total_local_filter_time += (t2 - t1);
    if (commGrid->GetRank() == 0) {
#ifdef MAGIQ_DEBUG
        cout << "\tlocal filter takes : " << (t2 - t1) << " s\n" << endl;
#endif
    }
}

// column based operation
void local_redistribution(PSpMat::MPI_DCCols &M, vector<IndexType> &range_table, int pair_size,
                          int pivot, vector<IndexType> &res) {
    double t1 = MPI_Wtime();

    res.clear();
    vector<IndexType>().swap(res);

    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    auto commGrid = M.getcommgrid();

    // get column offset and broadcast them
    int rowneighs = commGrid->GetGridCols();
    IndexType coffset[rowneighs + 1];

    int rowrank = commGrid->GetRankInProcRow();

    IndexType *locncols = new IndexType[rowneighs];  // number of rows is calculated by a reduction among the processor column
    locncols[rowrank] = M.getlocalcols();
    MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IndexType>(), locncols, 1, MPIType<IndexType>(), commGrid->GetRowWorld());
    coffset[rowrank] = std::accumulate(locncols, locncols + rowrank, static_cast<IndexType>(0));
    MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IndexType>(), coffset, 1, MPIType<IndexType>(), commGrid->GetRowWorld());
    delete[] locncols;

    // magic number for flag, useless
    coffset[rowneighs] = UINT32_MAX;

    // sort based on pivot
    double t1_sort_start = MPI_Wtime();
    symmergesort(range_table.data(), range_table.size() / pair_size, pair_size * sizeof(IndexType), comp[(pair_size - 3) * 5 + pivot]);
    double t1_sort_end = MPI_Wtime();

    vector<int> lens;
    lens.reserve(rowneighs + 1);
    lens.push_back(0);

    // find partial sum of sorted indices to ensure correct range
    int prev = 0;
    for (int i = 1; i <= rowneighs; ++i) {
        int j;
        for (j = prev; j < range_table.size() && range_table[j + pivot] < coffset[i]; j += pair_size) {}

        int len = j - prev;
        lens.push_back(len);
        prev = j;
    }

    vector<int> partial_sums(lens);
    partial_sum(lens.begin(), lens.end() - 1, partial_sums.begin());

    int *recvcount[rowneighs];
    vector<int> displs(rowneighs);
    displs[0] = 0;

    // only processors in first row enter here
    if (commGrid->GetRankInProcCol() == 0) {
        recvcount[myrank] = new int[rowneighs];

        // each process gathers recvcount from other processes
        for (int i = 0; i < rowneighs; ++i) {
            MPI_Gather(lens.data() + i + 1, 1, MPI_INT, recvcount[i], 1, MPI_INT, i, commGrid->GetRowWorld());
        }

        for (int j = 1; j < rowneighs; ++j) {
            displs[j] = displs[j - 1] + recvcount[myrank][j - 1];
        }

        res.resize(displs[rowneighs - 1] + recvcount[myrank][rowneighs - 1]);

        // gather real data
        for (int k = 0; k < rowneighs; ++k) {
            MPI_Gatherv(range_table.data() + partial_sums[k], lens[k + 1], MPIType<IndexType>(), res.data(),
                        recvcount[k], displs.data(), MPIType<IndexType>(), k, commGrid->GetRowWorld());
        }
    }

    // sort merged vector
    double t2_sort_start = MPI_Wtime();
    symmergesort(res.data(), res.size() / pair_size, pair_size * sizeof(IndexType), comp[(pair_size - 3) * 5 + pivot]);
    double t2_sort_end = MPI_Wtime();

    double t2 = MPI_Wtime();
    total_redistribution_time += (t2 - t1);
#ifdef MAGIQ_DEBUG
    if (myrank == 0) {
        cout << "\t sort 1 takes : " << (t1_sort_end - t1_sort_start) << " s" << endl;
        cout << "\t sort 2 takes : " << (t2_sort_end - t2_sort_start) << " s" << endl;
        cout << "\tlocal redistribution takes : " << (t2 - t1) << " s\n" << endl;
    }
#endif
}

void send_local_results(shared_ptr<CommGrid> commGrid, IndexType res_size) {
    int rowneighs = commGrid->GetGridRows();

    // process in first row
    if (commGrid->GetRank() < rowneighs) {
        if (commGrid->GetRank() != 0) {     // not myrank 0, send local result
            MPI_Send(&res_size, 1, MPIType<IndexType>(), 0, 0, commGrid->GetRowWorld());
        } else {                            // myrank 0, receive everything and output
            IndexType recv_size;
            for (IndexType i = 1; i < rowneighs; i++) {
                MPI_Recv(&recv_size, 1, MPIType<IndexType>(), i, 0, commGrid->GetRowWorld(), MPI_STATUS_IGNORE);
                res_size += recv_size;
            }
            cout << "final size : " << res_size << endl;
#ifdef MAGIQ_DEBUG
            cout << "total get local indices time : " << total_get_local_indices_time << " s" << endl;
            cout << "total send local indices time : " << total_send_local_indices_time << " s" << endl;
            cout << "total local join time : " << total_local_join_time << " s" << endl;
            cout << "total local filter time : " << total_local_filter_time << " s" << endl;
            cout << "total local redistribution time : " << total_redistribution_time << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
#endif
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////
//                           Sparsql Parse Phase                                //
//////////////////////////////////////////////////////////////////////////////////

// default line buffer size, 100
static const size_t lineSize = 100;

int parseLine(string &line, map<string, PSpMat::MPI_DCCols> &matrices, 
        map<string, FullyDistVec<IndexType, ElementType> > vectors,
        PSpMat::MPI_DCCols &G, FullyDistVec<IndexType, ElementType> &dm,
        bool isPerm, FullyDistVec<IndexType, IndexType> &nonisov) {
    
    // get common world
    auto commWorld = G.getcommgrid();
    int myrank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    
    // remove all spaces in the string
    line.erase(remove(line.begin(), line.end(), ' '), line.end());
    // if the last character is new line, then remove it
    if (line.back() == '\n')
        line.pop_back();
    // if (myrank == 0)
    //     cout << line << endl;

    // find the assignment operation
    int eqOp = line.find('=');
    if (eqOp == string::npos) {
        // there is no '=', error
    }

    // sermiring multiplication, by default is true
    bool isSermiring = true;
    // columnDiag = true : Column in diagonalizeV
    // columnDiag = false : Row in diagonalizeV
    bool columnDiag = true;
    // scale for diagonalizeV
    ElementType scale = 1;
    // columnApply = true : Column in multDimApplyPrune
    // columnApplu = false : Row in multDimApplyPrune
    bool columnApply = true;
    // this is used only when G.T is used
    bool afterTranspose = false;
    
    // ⊗ : 3 chars
    // × : 2 chars
    int multOp = line.find("⊗");
    if (multOp == -1) {
        multOp = line.find("×");
        isSermiring = false;
    }

    // cout << eqOp << "\t" << multOp << endl;

    string interMat = line.substr(0, eqOp);
    string mult1 = line.substr(eqOp + 1, multOp - eqOp - 1);
    string mult2 = line.substr(multOp + 2 + isSermiring);

    // if (myrank == 0) {
    //     cout << interMat << "\t" << mult1 << "\t" << mult2 << endl;
    // }

    // interMat is always m_x_x, there is no need to parse it

    // start parsing from mult1
    if (mult1[0] == 'G') {  // G or G.T
        if (mult1 == "G") {
            matrices[interMat] = PSpMat::MPI_DCCols(G);
        } else if (mult1 == "G.T") {
            matrices[interMat] = PSpMat::MPI_DCCols(G);
            columnApply = false;
            afterTranspose = true;
        } else {
            // error, only G and G.T are accepted here
        }

        // parse mult2
        if (mult2[0] == 'I') {
            int dimOp = mult2.find('^');
            if (dimOp == string::npos) {
                // there must be ^ to indicate where the value is
            }
            int scaleOp = mult2.find('*');
            int pos = atoi(mult2.substr(dimOp + 1, scaleOp - dimOp - 1).c_str());
            
            if (scaleOp != string::npos) {
                scale *= atoi(mult2.substr(scaleOp + 1).c_str());
            }

            FullyDistVec<IndexType, ElementType> rt(commWorld, G.getnrow(), 0);
            IndexType ind = nonisov[pos];
            rt.SetElement(ind, scale);

            // parameters : matrix *, fullyvec, columnApply, sermiring
            multDimApplyPrune(matrices[interMat], rt, columnApply ? Column : Row, isSermiring);
            if (afterTranspose)
                matrices[interMat].Transpose();
            return matrices[interMat].getnnz();

        } else if (mult2[0] == 'm') {
            int dot1 = mult2.find('.');
            int dot2 = mult2.rfind('.');
            int scaleOp = mult2.find('*');

            string mat = mult2.substr(0, dot1);
            if (dot1 != string::npos) {
                if (dot2 != dot1) {
                    // fst : T, snd : D
                    string fst = mult2.substr(dot1 + 1, dot2 - dot1 - 1);
                    string snd = mult2.substr(dot2 + 1, scaleOp != string::npos?scaleOp - dot2 - 1:scaleOp);
                    if (fst.compare("T") != 0 || snd.compare("D") != 0) {
                        // error, m_x_x.T.D is only one acceptable
                    }
                    // m_x_x.T.D
                } else {
                    // fst : T or D
                    string fst = mult2.substr(dot1 + 1, scaleOp != string::npos?scaleOp - dot1 - 1:scaleOp);
                    if (fst.compare("T") == 0) {
                        // error, should be D always
                    } else
                        columnDiag = false;
                }
            } else {
                // error, there shoulw be dot1
            }

            if (scaleOp != string::npos) {
                scale *= atoi(mult2.substr(scaleOp + 1).c_str());
            }

            diagonalizeV(matrices[mat], dm, columnDiag ? Column : Row, scale);
            multDimApplyPrune(matrices[interMat], dm, columnApply ? Column : Row, isSermiring);
            if (afterTranspose)
                matrices[interMat].Transpose();
            return matrices[interMat].getnnz();

        } else {
            // error, only I^xxxx*xxxx and m_x_x(.T).D*xxxx are good
        }

    } else if (mult1[0] == 'I') {   // I^xxxx*xxxx
        // mult2 should be the same as interMat
        if (interMat == mult2) {
            columnApply = false;
            isSermiring = false;

            int dimOp = mult1.find('^');
            if (dimOp == string::npos) {
                // there must be ^ to indicate where the value is
            }
            int scaleOp = mult1.find('*');
            int pos = atoi(mult1.substr(dimOp + 1, scaleOp - dimOp - 1).c_str());
            
            if (scaleOp != string::npos) {
                scale *= atoi(mult1.substr(scaleOp + 1).c_str());
            }

            FullyDistVec<IndexType, ElementType> rt(commWorld, G.getnrow(), 0);
            IndexType ind = nonisov[pos];
            rt.SetElement(ind, scale);

            multDimApplyPrune(matrices[interMat], rt, columnApply ? Column : Row, isSermiring);
            if (afterTranspose)
                matrices[interMat].Transpose();
            return matrices[interMat].getnnz();

        } else {
            // error, if mult1 is I, then should have mult2 == interMat
        }

    } else if (mult1[0] == 'm') {   // m_x_x, m_x_x.T, m_x_x.D*xxx or m_x_x.T.D*xxxx
        if (mult2 == interMat) {
            columnApply = false;

            int dot1 = mult1.find('.');
            int dot2 = mult1.rfind('.');

            if (dot2 != dot1) {
                string fst = mult1.substr(dot1 + 1, dot2 - dot1 - 1);
                string snd = mult1.substr(dot2 + 1);
                if (fst.compare("T") != 0 || snd.compare("D") != 0) {
                    // error, m_x_x.T.D is only one acceptable
                }
            } else {
                string fst = mult1.substr(dot1 + 1);
                if (fst.compare("D") == 0) {
                    // error, if only one dot found, then it should be D
                }
                columnDiag = false;
            }

            string mat = mult1.substr(0, dot1);
            diagonalizeV(matrices[mat], dm, columnDiag ? Column : Row, scale);
            multDimApplyPrune(matrices[interMat], dm, columnApply ? Column : Row, isSermiring);
            if (afterTranspose)
                matrices[interMat].Transpose();
            return matrices[interMat].getnnz();

        } else if (mult1 == interMat) {
            int dot1 = mult2.find('.');
            int dot2 = mult2.rfind('.');

            if (dot2 != dot1) {
                string fst = mult2.substr(dot1 + 1, dot2 - dot1 - 1);
                string snd = mult2.substr(dot2 + 1);
                if (fst.compare("T") != 0 || snd.compare("D") != 0) {
                    // error, m_x_x.T.D is only one acceptable
                }
            } else {
                string fst = mult2.substr(dot1 + 1);
                if (fst.compare("D") == 0) {
                    // error, if only one dot found, then it should be D
                }
                columnDiag = false;
            }

            string mat = mult2.substr(0, dot1);
            diagonalizeV(matrices[mat], dm, columnDiag ? Column : Row, scale);
            multDimApplyPrune(matrices[interMat], dm, columnApply ? Column : Row, isSermiring);
            if (afterTranspose)
                matrices[interMat].Transpose();
            return matrices[interMat].getnnz();

        } else {
            // error, mult1 = interMat or mult2 = interMat
        }
    } else {
        // error, there should be one of (mult1, mult2) that equals to interMat
    }
    // end of function call
}

int parseSparql(const char* sparqlFile, 
        map<string, PSpMat::MPI_DCCols> &matrices, 
        map<string, FullyDistVec<IndexType, ElementType> > &vectors,
        PSpMat::MPI_DCCols &G, FullyDistVec<IndexType, ElementType> &dm,
        bool isPerm, FullyDistVec<IndexType, IndexType> &nonisov) {
    // get common world
    auto commWorld = G.getcommgrid();
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    //open and get the file handle
    FILE *fh;
    fh = fopen(sparqlFile, "r");

    //check if file exists
    if (fh == NULL){
        if (myrank == 0) {
            printf("file does not exists : %s\n", sparqlFile);
            return 0;
        }
    }

    if (myrank == 0) {
        printf("Start reading sparql file ...\n");
    }

    // line buffer
    char* line = (char *)malloc(lineSize);

    int cntLines = 1;
    while (fgets(line, lineSize, fh) != NULL) {
        string str(line);
        if (myrank == 0) {
            if (cntLines == 1) {
                cout << "###############################################################" << endl << flush;
                cout << "Start Running query..." << endl << flush;
                cout << "###############################################################" << endl << flush;
            }
            cout << "---------------------------------------------------------------" << endl << flush;
            cout << "step " << cntLines << " : " << str << endl << flush;
        }
        int ret = parseLine(str, matrices, vectors, G, dm, isPerm, nonisov);
        if (myrank == 0) {
            cout << "---------------------------------------------------------------" << endl << flush;
        }

        if (ret == 0) {
            // early stop
            return 0;
        } else if (ret < 0) {
            // error happened
            return -1;
        }
        cntLines++;
    }

    // free momery
    free(line);

    return 0;
}

#endif //COMBINATORIAL_BLAS_HEADER_SCAL_H
