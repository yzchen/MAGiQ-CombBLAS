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

// utility definitions 
#include "utils.h"

using namespace std;
using namespace combblas;

// Sparse matrix definition
class PSpMat {
public:
    typedef SpDCCols<IndexType, ElementType> DCCols;
    typedef SpParMat<IndexType, ElementType, DCCols> MPI_DCCols;
};

//////////////////////////////////////////////////////////////////////////////////
//                           Graph Loading Phase                                //
//////////////////////////////////////////////////////////////////////////////////

// handle duplicate in original loaded data
ElementType selectSecond(ElementType a, ElementType b) { return b; }

// permute input matrix G randomly, permutation infotmation will be stored in nonisov
// this function is never called because now permutation is implemented during data loading
void permute(PSpMat::MPI_DCCols &G, FullyDistVec<IndexType, IndexType> &nonisov) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // permute G
    double t_perm1 = MPI_Wtime();

    nonisov.iota(G.getnrow(), 0);
    nonisov.RandPerm();

    // permute G in place
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

//////////////////////////////////////////////////////////////////////////////////
//                          Query Execution Phase                               //
//////////////////////////////////////////////////////////////////////////////////

// time for running queries
static double total_reduce_time = 0.0;
static double total_prune_time = 0.0;
static double total_mmul_scalar_time = 0.0;
static double total_dim_apply_time = 0.0;

// reset all time reslated to query execution
void clear_query_time() {
    total_reduce_time = 0.0;
    total_prune_time = 0.0;
    total_mmul_scalar_time = 0.0;
    total_dim_apply_time = 0.0;
}

bool isZero(ElementType t) { return t == 0; }

// special rdf semiring for dimApply
ElementType rdf_multiply(ElementType a, ElementType b) {
    if (a != 0 && b != 0 && a == b) {   return static_cast<ElementType>(1);
                            } else {    return static_cast<ElementType>(0);     }
}

// diagonalize(reduce) based on dim(Row/Column, default is Row), 
// then scale the FullyDistVec with scalar(default is 1)
// output : diag
void diagonalizeV(const PSpMat::MPI_DCCols &M, FullyDistVec<IndexType, ElementType> &diag, 
                    Dim dim = Row, ElementType scalar = 1) {
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

// dimApply via dim(Row/Column) with sermiring(isRDF, only two sermirings) to matrix A, 
// change matrix A in place
// then prune 0s in the result matrix
void multDimApplyPrune(PSpMat::MPI_DCCols &A, FullyDistVec<IndexType, ElementType> &v, 
                        Dim dim, bool isRDF) {
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
    // printInfo after each computation step
    A.PrintInfo();

#ifdef MAGIQ_DEBUG
    if (myrank == 0) {
        cout << "\tdim-apply takes: " << (t2 - t1) << " s" << endl;
        cout << "\tprune takes: " << (t4 - t3) << " s" << endl;
    }
#endif
}

//////////////////////////////////////////////////////////////////////////////////
//                          Result Generation Phase                             //
//////////////////////////////////////////////////////////////////////////////////

// time for result generation
static double total_get_local_indices_time = 0.0;
static double total_send_local_indices_time = 0.0;
static double total_local_join_time = 0.0;
static double total_local_filter_time = 0.0;
static double total_redistribution_time = 0.0;

// reset all time related to result generation
void clear_result_time() {
    total_get_local_indices_time = 0.0;
    total_send_local_indices_time = 0.0;
    total_local_join_time = 0.0;
    total_local_filter_time = 0.0;
    total_redistribution_time = 0.0;
}

// M should have same rows and cols
// after this function call, indices will be filled, 
// indices will have size should be even, I and J are together
void get_local_indices(PSpMat::MPI_DCCols &M, vector<IndexType> &indices) {
    double t_start = MPI_Wtime();
    assert(M.getnrow() == M.getncol());

    indices.clear();

    auto commGrid = M.getcommgrid();
    int colrank = commGrid->GetRankInProcCol();
    int rowrank = commGrid->GetRankInProcRow();

    // calculate row offset for each process
    int colneighs = commGrid->GetGridRows();
    // number of rows is calculated by a reduction among the processor column
    IndexType *locnrows = new IndexType[colneighs];  
    locnrows[colrank] = M.getlocalrows();
    MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IndexType>(), locnrows, 1, MPIType<IndexType>(), commGrid->GetColWorld());
    IndexType roffset = std::accumulate(locnrows, locnrows + colrank, static_cast<IndexType>(0));
    delete[] locnrows;

    // calculate column offset for each process
    int rowneighs = commGrid->GetGridCols();
    // number of columns is calculated by a reduction among the processor row
    IndexType *locncols = new IndexType[rowneighs];  
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
        assert(indices.size() % 2 == 0 && "indices should have even size!");
    }

    double t_end = MPI_Wtime();
#ifdef MAGIQ_DEBUG
    if (commGrid->GetRank() == 0) {
        cout << "\tget local indices takes : " << (t_start - t_end) << " s" << endl;
    }
#endif
    total_get_local_indices_time += (t_start - t_end);
}

// first has range [l1, r1), step size pair_size1
// second has range [l2, r2), step size pair_size2
// merge sort should be based on pivot1 and pivot2(they are both offset, 0 - pair_size1/pair_size2)
// output : first, so first should have enough space for merged vector
void merge_local_vectors(vector<IndexType> &first, vector<IndexType> &second, 
                        IndexType l1, IndexType l2, IndexType r1, IndexType r2, 
                        int pair_size1, int pair_size2, int pivot1, int pivot2) {
    IndexType i = l1, j = l2;

    vector<IndexType> res;
    res.reserve((r1 - l1) + (r2 - l2));

    auto it1 = first.begin(), it2 = second.begin();
    while (i < r1 && j < r2) {
        if (first[i + pivot1] <= second[j + pivot2]) {
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
// (first row will work in the following)
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
                MPI_Recv(recv_I.data(), number_count, MPIType<IndexType>(), sender, 0, 
                    commGrid->GetColWorld(), MPI_STATUS_IGNORE);
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

// push one entry to the vector res, 
// entry information comes from source1 and source2, they are one of inner join results
// index1 is the position of source1, index2 is the position of source2,
// push should follow the order specified in vector order 
void put_tuple(vector<IndexType> &res, vector<IndexType> &source1, vector<IndexType> &source2,
               IndexType index1, IndexType index2, vector<IndexType> &order) {
    for (IndexType oi = 0; oi < order.size(); oi += 2) {
        // order[oi] == 0 or 1, other choices will be treated as 1
        if (order[oi] == 0) {   res.push_back(source1[index1 + order[oi + 1]]);
        } else  {   res.push_back(source2[index2 + order[oi + 1]]); }
    }
}

// local join with special tables(indices1 and indices2), only processors in first row work
// both indices1 and indices2 are seriliazed, have actual size n * pair_size
// pivot1 and pivot2 are main keys for joining
// order : size * 2
// 0 0 0 1 1 0 means in result, first column is col 0 in index 0, then col 1 in index 0, 
// the last one is col 0 in index 1
// this also means joining is based on col 0 in index 0 and col 0 in index 1
// output : res should be different from indices1 and indices2
void local_join(shared_ptr<CommGrid> commGrid, vector<IndexType> &indices1, vector<IndexType> &indices2, 
                int pair_size1, int pair_size2, int pivot1, int pivot2, 
                vector<IndexType> &order, vector<IndexType> &res) {
    double t1 = MPI_Wtime();

    res.clear();
    res.shrink_to_fit();

    if (commGrid->GetRankInProcCol() == 0) {        // first row processors
        IndexType i1 = 0, i2 = 0;
        IndexType sz1 = indices1.size(), sz2 = indices2.size();

        while (i1 < sz1 && i2 < sz2) {
            if (indices1[i1 + pivot1] < indices2[i2 + pivot2]) {
                i1 += pair_size1;
            } else if (indices1[i1 + pivot1] > indices2[i2 + pivot2]) {
                i2 += pair_size2;
            } else {
                put_tuple(res, indices1, indices2, i1, i2, order);

                IndexType i22 = i2 + pair_size2;
                while (i22 < sz2 && indices1[i1 + pivot1] == indices2[i22 + pivot2]) {
                    put_tuple(res, indices1, indices2, i1, i22, order);
                    i22 += pair_size2;
                }

                IndexType i11 = i1 + pair_size1;
                while (i11 < sz1 && indices1[i11 + pivot1] == indices2[i2 + pivot2]) {
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

// local filter with special tables(indices1 and indices2), only processors in first row work
// both indices1 and indices2 are seriliazed, have actual size n * pair_size
// pivot11 and pivot21 are main keys for filter, tables should be sorted based on them
// pivot12 and pivot22 are renaming equivalent
// output : res should be different from indices1 and indices2
void local_filter(shared_ptr<CommGrid> commGrid, vector<IndexType> &indices1, vector<IndexType> &indices2, 
                    int pair_size1, int pair_size2, int pivot11, int pivot12, int pivot21, int pivot22, 
                    vector<IndexType> &order, vector<IndexType> &res) {
    double t1 = MPI_Wtime();

    res.clear();
    res.reserve(2000);      // 2000 is a magic number, just to somehow reserve memory

    if (commGrid->GetRankInProcCol() == 0) {
        IndexType i1 = 0, i2 = 0;
        IndexType sz1 = indices1.size(), sz2 = indices2.size();

        while (i1 < sz1 && i2 < sz2) {
            if (indices1[i1 + pivot11] < indices2[i2 + pivot21]) {
                i1 += pair_size1;
            } else if (indices1[i1 + pivot11] > indices2[i2 + pivot21]) {
                i2 += pair_size2;
            } else {
                if (indices1[i1 + pivot12] == indices2[i2 + pivot22]) {
                    put_tuple(res, indices1, indices2, i1, -1, order);
                }

                IndexType i22 = i2 + pair_size2;
                while (i22 < sz2 && indices1[i1 + pivot11] == indices2[i22 + pivot21]) {
                    if (indices1[i1 + pivot12] == indices2[i22 + pivot22]) {
                        put_tuple(res, indices1, indices2, i1, -1, order);
                    }
                    i22 += pair_size2;
                }

                IndexType i11 = i1 + pair_size1;
                while (i11 < sz1 && indices1[i11 + pivot11] == indices2[i2 + pivot21]) {
                    if (indices1[i11 + pivot12] == indices2[i2 + pivot22]) {
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

// redistributed the table(range_table with pair_size) based on the matrix M
// column based operation
// output : res 
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

    // number of rows is calculated by a reduction among the processor column
    IndexType *locncols = new IndexType[rowneighs];  
    locncols[rowrank] = M.getlocalcols();
    MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IndexType>(), locncols, 1, MPIType<IndexType>(), commGrid->GetRowWorld());
    coffset[rowrank] = std::accumulate(locncols, locncols + rowrank, static_cast<IndexType>(0));
    MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IndexType>(), coffset, 1, MPIType<IndexType>(), commGrid->GetRowWorld());
    delete[] locncols;

    // magic number for flag, useless
    coffset[rowneighs] = UINT32_MAX;

    // sort based on pivot
    double t1_sort_start = MPI_Wtime();
    symmergesort(range_table.data(), range_table.size() / pair_size, 
        pair_size * sizeof(IndexType), comp[(pair_size - 3) * 5 + pivot]);
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

// rank 0 will have final correct result, other ranks have 0
int send_local_results(shared_ptr<CommGrid> commGrid, IndexType res_size) {
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
#ifdef MAGIQ_DEBUG
            cout << "step size : " << res_size << endl;
            cout << "total accumulated get local indices time : " << total_get_local_indices_time << " s" << endl;
            cout << "total accumulated send local indices time : " << total_send_local_indices_time << " s" << endl;
            cout << "total accumulated local join time : " << total_local_join_time << " s" << endl;
            cout << "total accumulated local filter time : " << total_local_filter_time << " s" << endl;
            cout << "total accumulated local redistribution time : " << total_redistribution_time << " s" << endl;
#endif
        }
    }

    return res_size;
}

//////////////////////////////////////////////////////////////////////////////////
//                           Sparsql Parse Phase                                //
//////////////////////////////////////////////////////////////////////////////////

// default line buffer size, 100
static const size_t lineSize = 100;

// matrices have all the intermediate matrices
// G is input data graph, dm will be reused for all cases
// isPerm denotes if G is permutated, nonisov is permutation vector
// output : nnz of the results
int parseQueryLine(string &line, map<string, PSpMat::MPI_DCCols> &matrices, 
        PSpMat::MPI_DCCols &G, FullyDistVec<IndexType, ElementType> &dm,
        bool isPerm, FullyDistVec<IndexType, IndexType> &nonisov, int cntLines) {
    
    // get common world
    auto commGrid = G.getcommgrid();
    int myrank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    
    // remove all spaces in the string
    line.erase(remove(line.begin(), line.end(), ' '), line.end());
    // if the last character is new line, then remove it
    if (line.back() == '\n')
        line.pop_back();

    // find the assignment operation
    int eqOp = line.find('=');
    if (eqOp == string::npos) {
        // there is no '=', error
        return -1;
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

    string interMat = line.substr(0, eqOp);
    string mult1 = line.substr(eqOp + 1, multOp - eqOp - 1);
    string mult2 = line.substr(multOp + 2 + isSermiring);

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
            return -1;
        }

        // parse mult2
        if (mult2[0] == 'I') {
            int dimOp = mult2.find('^');
            if (dimOp == string::npos) {
                // error, there must be ^ to indicate where the value is
                return -1;
            }
            int scaleOp = mult2.find('*');
            int pos = atoi(mult2.substr(dimOp + 1, scaleOp - dimOp - 1).c_str());
            
            if (scaleOp != string::npos) {
                scale *= atoi(mult2.substr(scaleOp + 1).c_str());
            }

            FullyDistVec<IndexType, ElementType> rt(commGrid, G.getnrow(), 0);
            if (isPerm) {
                IndexType ind = nonisov[pos];
                rt.SetElement(ind, scale);
            } else {
                rt.SetElement(pos, scale);
            }

            double t_start = MPI_Wtime();
            // parameters : matrix *, fullyvec, columnApply, sermiring
            multDimApplyPrune(matrices[interMat], rt, columnApply ? Column : Row, isSermiring);
            if (afterTranspose)
                matrices[interMat].Transpose();
            double t_end = MPI_Wtime();
            if (myrank == 0) {
                cout << "query step time (" << cntLines << ") : " <<  (t_end - t_start) << " s" << endl;
            }
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
                        return -1;
                    }
                    // m_x_x.T.D
                } else {
                    // fst : T or D
                    string fst = mult2.substr(dot1 + 1, scaleOp != string::npos?scaleOp - dot1 - 1:scaleOp);
                    if (fst.compare("T") == 0) {
                        // error, should be D always
                        return -1;
                    } else
                        columnDiag = false;
                }
            } else {
                // error, there shoulw be dot1
                return -1;
            }

            if (scaleOp != string::npos) {
                scale *= atoi(mult2.substr(scaleOp + 1).c_str());
            }

            double t_start = MPI_Wtime();
            diagonalizeV(matrices[mat], dm, columnDiag ? Column : Row, scale);
            multDimApplyPrune(matrices[interMat], dm, columnApply ? Column : Row, isSermiring);
            if (afterTranspose)
                matrices[interMat].Transpose();
            double t_end = MPI_Wtime();
            if (myrank == 0) {
                cout << "query step time (" << cntLines << ") : " <<  (t_end - t_start) << " s" << endl;
            }
            return matrices[interMat].getnnz();

        } else {
            // error, only I^xxxx*xxxx and m_x_x(.T).D*xxxx are good
            return -1;
        }

    } else if (mult1[0] == 'I') {   // I^xxxx*xxxx
        // mult2 should be the same as interMat
        if (interMat == mult2) {
            columnApply = false;
            isSermiring = false;

            int dimOp = mult1.find('^');
            if (dimOp == string::npos) {
                // there must be ^ to indicate where the value is
                return -1;
            }
            int scaleOp = mult1.find('*');
            int pos = atoi(mult1.substr(dimOp + 1, scaleOp - dimOp - 1).c_str());
            
            if (scaleOp != string::npos) {
                scale *= atoi(mult1.substr(scaleOp + 1).c_str());
            }

            FullyDistVec<IndexType, ElementType> rt(commGrid, G.getnrow(), 0);
            IndexType ind = nonisov[pos];
            if (isPerm) {
                IndexType ind = nonisov[pos];
                rt.SetElement(ind, scale);
            } else {
                rt.SetElement(pos, scale);
            }

            double t_start = MPI_Wtime();
            multDimApplyPrune(matrices[interMat], rt, columnApply ? Column : Row, isSermiring);
            if (afterTranspose)
                matrices[interMat].Transpose();
            double t_end = MPI_Wtime();
            if (myrank == 0) {
                cout << "query step time (" << cntLines << ") : " <<  (t_end - t_start) << " s" << endl;
            }
            return matrices[interMat].getnnz();

        } else {
            // error, if mult1 is I, then should have mult2 == interMat
            return -1;
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
                    return -1;
                }
            } else {
                string fst = mult1.substr(dot1 + 1);
                if (fst.compare("D") != 0) {
                    // error, if only one dot found, then it should be D
                    return -1;
                }
                columnDiag = false;
            }

            string mat = mult1.substr(0, dot1);
            double t_start = MPI_Wtime();
            diagonalizeV(matrices[mat], dm, columnDiag ? Column : Row, scale);
            multDimApplyPrune(matrices[interMat], dm, columnApply ? Column : Row, isSermiring);
            if (afterTranspose)
                matrices[interMat].Transpose();
            double t_end = MPI_Wtime();
            if (myrank == 0) {
                cout << "query step time (" << cntLines << ") : " <<  (t_end - t_start) << " s" << endl;
            }
            return matrices[interMat].getnnz();

        } else if (mult1 == interMat) {
            int dot1 = mult2.find('.');
            int dot2 = mult2.rfind('.');

            if (dot2 != dot1) {
                string fst = mult2.substr(dot1 + 1, dot2 - dot1 - 1);
                string snd = mult2.substr(dot2 + 1);
                if (fst.compare("T") != 0 || snd.compare("D") != 0) {
                    // error, m_x_x.T.D is only one acceptable
                    return -1;
                }
            } else {
                string fst = mult2.substr(dot1 + 1);
                if (fst.compare("D") != 0) {
                    // error, if only one dot found, then it should be D
                    return -1;
                }
                columnDiag = false;
            }

            string mat = mult2.substr(0, dot1);
            double t_start = MPI_Wtime();
            diagonalizeV(matrices[mat], dm, columnDiag ? Column : Row, scale);
            multDimApplyPrune(matrices[interMat], dm, columnApply ? Column : Row, isSermiring);
            if (afterTranspose)
                matrices[interMat].Transpose();
            double t_end = MPI_Wtime();
            if (myrank == 0) {
                cout << "query step time (" << cntLines << ") : " <<  (t_end - t_start) << " s" << endl;
            }
            return matrices[interMat].getnnz();

        } else {
            // error, mult1 = interMat or mult2 = interMat
            return -1;
        }
    } else {
        // error, there should be one of (mult1, mult2) that equals to interMat
        return -1;
    }
    // end of function call
}

// labels should be ordered
// indl will be updated after this function call(join/filter)
// output : nnz in the result vector
int parseResGenLine(shared_ptr<CommGrid> commGrid, string &line, 
        map<string, PSpMat::MPI_DCCols> &matrices, 
        vector<char> &labels, char &primary, vector<IndexType> &indl, int cntLines) {

    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (line.back() == '\n')
        line.pop_back();

    int splitOp = line.find('%');
    int eqOp = line.rfind('%');
    if (splitOp != string::npos) {
        string opType = line.substr(0, splitOp);
        string mat;
        bool isJoin = true && (opType.compare("join") == 0);

        if (eqOp == splitOp)        // if it is join
            mat = line.substr(splitOp + 1);
        else {        // if it is filter
            mat = line.substr(eqOp + 1);
            if (opType.compare("filter") == 0)
                isJoin = false;
            else {
                // error, only join and filter can be accepted
                return 0;
            }
        }

        int dash1 = mat.find('_');
        int dash2 = mat.rfind('_');
        if (dash1 == string::npos || dash1 == dash2) {
            // error, mat name should be m_x_x
            return 0;
        }

        if (labels.empty()) {   // first join, only one matrix is avaible now, use transpose of it, initialize labels
            labels.push_back(mat[dash2 + 1]);
            labels.push_back(mat[dash1 + 1]);

            primary = mat[dash2 + 1];

            double t_start = MPI_Wtime();
            matrices[mat].Transpose();
            get_local_indices(matrices[mat], indl);
            send_local_indices(commGrid, indl);
            matrices[mat].FreeMemory();
            double t_end = MPI_Wtime();
            if (myrank == 0) {
                cout << "result generation step (" << cntLines << ") : " << (t_end - t_start) << " s" << endl;
            }

            return send_local_results(commGrid, indl.size() / labels.size());
        } else {
            int nres;

            vector<IndexType> indj, indr, indw;

            double t_start = MPI_Wtime();
            get_local_indices(matrices[mat], indj);
            send_local_indices(commGrid, indj);

            // send_local_results(commGrid, indj.size() / labels.size());

            // determine which two columns are corresponded
            // pivot2 should be always 1, use the second column to reduce
            int pivot1 = -1, pivot2 = 1;
            auto v2 = std::find(labels.begin(), labels.end(), mat[dash2 + 1]);
            if (v2 != labels.end()) {
                pivot1 = v2 - labels.begin();
            } else {
                // error, there should be key always
                return 0;
            }
            // determine which two columns are corresponded
            
            if (labels.size() > 2 && primary != mat[dash2 + 1]) {
                local_redistribution(matrices[mat], indl, labels.size(), pivot1, indw);
            } else
                indw = indl;

            matrices[mat].FreeMemory();

            vector<IndexType> orders(labels.size() * 2);
            for (int i = 0; i < labels.size(); i++) {
                orders[2 * i] = 0;
                orders[2 * i + 1] = i;
            }

            // default value is the end of labels vector
            // so when you don't find any value larger then mat[dash1 + 1], that means should insert at the end
            int insert_pos = labels.size();
            if (isJoin) {
                for (int i = 0; i < labels.size(); i++)
                    if (labels[i] > mat[dash1 + 1]) {
                        insert_pos = i;
                        break;
                    }

                orders.insert(orders.begin() + 2 * insert_pos, 1);
                orders.insert(orders.begin() + 2 * insert_pos + 1, 0);
            }

            if (isJoin) {
                local_join(commGrid, indw, indj, labels.size(), 2, pivot1, 1, orders, indr);
                labels.insert(labels.begin() + insert_pos, mat[dash1 + 1]);
                double t_end = MPI_Wtime();
                if (myrank == 0) {
                    cout << "result generation step (" << cntLines << ") : " << (t_end - t_start) << " s" << endl;
                }
                nres = send_local_results(commGrid, indr.size() / labels.size());
            } else {
                int renameInd = std::find(labels.begin(), labels.end(), line[eqOp - 1]) - labels.begin();
                local_filter(commGrid, indw, indj, labels.size(), 2, pivot1, renameInd, 1, 0, orders, indr);
                double t_end = MPI_Wtime();
                if (myrank == 0) {
                    cout << "result generation step (" << cntLines << ") : " << (t_end - t_start) << " s" << endl;
                }
                nres = send_local_results(commGrid, indr.size() / labels.size());
            }

            indl = indr;
            if (myrank == 0) {
                cout << "===> step result size : " << nres << endl;
            }
            return nres;
        }
    }
    return 0;
}

// matrices have all the intermediate matrices
// G is input data graph, dm will be reused for all cases
// isPerm denotes if G is permutated, nonisov is permutation vector
int parseSparql(const char* sparqlFile, 
        map<string, PSpMat::MPI_DCCols> &matrices, 
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

    int qres = 0, rres = 0;
    // line buffer
    char* line = (char *)malloc(lineSize);
    double t_start = MPI_Wtime();
    if (fgets(line, lineSize, fh) != NULL) {
        string str(line);
        if (str.back() == '\n')
        str.pop_back();
        if (myrank == 0) {
            cout << "###############################################################" << endl << flush;
            cout << "Start Running query " << str << "..." << endl << flush;
            cout << "###############################################################" << endl << flush;
        }

        clear_query_time();
        clear_result_time();

        bool execute = true;
        int cntLines = 1;
        vector<IndexType> indl;
        char primary;
        vector<char> labels;
        while (fgets(line, lineSize, fh) != NULL) {
            string str(line);
            if (str.compare("query_execution\n") == 0) {
                execute = true;
            } else if (str.compare("result_generation\n") == 0) {
                cntLines = 1;
                execute = false;
            } else {
                if (execute) {      // query execution
                    if (myrank == 0) {
                        cout << "---------------------------------------------------------------" << endl << flush;
                        cout << "query execute step " << cntLines << " : " << str << endl << flush;
                    }
                    qres = parseQueryLine(str, matrices, G, dm, isPerm, nonisov, cntLines);

                    if (qres == 0) {
                        // early stop
                        return 0;
                    } else if (qres < 0) {
                        // error happened
                        return -1;
                    }

                } else {    // result generation
                    if (myrank == 0) {
                        cout << "---------------------------------------------------------------" << endl << flush;
                        cout << "result generate step " << cntLines << " : " << str << endl << flush;
                    }
                    rres = parseResGenLine(commWorld, str, matrices, labels, primary, indl, cntLines);
                }
                cntLines++;
            }
        }
    }
    double t_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "---------------------------------------------------------------" << endl << flush;
        cout << "final result : " << rres << endl;
        cout << "total query time : " << (t_end - t_start) << " s" << endl;
        cout << "###############################################################" << endl << flush;
    }

    // free momery
    free(line);

    return 0;
}

#endif //COMBINATORIAL_BLAS_HEADER_SCAL_H
