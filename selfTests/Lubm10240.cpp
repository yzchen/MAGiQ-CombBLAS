#include <iostream>
#include <functional>
#include <algorithm>
#include <sstream>
#include <vector>
#include <iterator>
#include <fstream>
#include "../include/Header10240.h"
#include "../include/CombBLAS.h"

using namespace std;
using namespace combblas;

// for constructing diag matrix
static FullyDistVec<IndexType, ElementType> *rvec;
static FullyDistVec<IndexType, ElementType> *qvec;

void resgen_l1(PSpMat::MPI_DCCols &m_50, PSpMat::MPI_DCCols &m_35, PSpMat::MPI_DCCols &m_43, PSpMat::MPI_DCCols &m_64,
               PSpMat::MPI_DCCols &m_24, PSpMat::MPI_DCCols &m_13) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t1_t = MPI_Wtime();
    total_get_local_indices_time = 0.0;
    total_send_local_indices_time = 0.0;
    total_local_join_time = 0.0;
    total_local_filter_time = 0.0;
    total_redistribution_time = 0.0;
    total_send_result_time = 0.0;

    // m_50 becoms m_05
    m_50.Transpose();

    auto commGrid = m_50.getcommgrid();

    vector<IndexType> index_05, index_35, index_035_0, index_035, index_43, index_0345_0, index_0345,
            index_64, index_03456, index_24, index_023456_0, index_023456, index_13, index_0123456;

    vector<IndexType> order1 = {0, 0, 1, 0, 1, 1}, order2 = {0, 0, 0, 1, 1, 0, 0, 2}, order3 = {0, 0, 0, 1, 0, 2, 0, 3},
            order4 = {0, 0, 1, 0, 0, 1, 0, 2, 0, 3}, order5 = {0, 0, 1, 0, 0, 1, 0, 2, 0, 3, 0, 4};
    double t2_t = MPI_Wtime();

    if (myrank == 0) {
        cout << "begin result generation ......" << endl;
        cout << "\ttranspose matrix and declarations take : " << (t2_t - t1_t) << " s\n" << endl;
    }

    double t2_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    get_local_indices(m_50, index_05);
    MPI_Barrier(MPI_COMM_WORLD);
    double t2_end = MPI_Wtime();
    double t2 = t2_end - t2_start;

    double t3_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    send_local_indices(commGrid, index_05);
    MPI_Barrier(MPI_COMM_WORLD);
    double t3_end = MPI_Wtime();
    double t3 = t3_end - t3_start;

    double t4_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    get_local_indices(m_35, index_35);
    MPI_Barrier(MPI_COMM_WORLD);
    double t4_end = MPI_Wtime();
    double t4 = t4_end - t4_start;

    double t5_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    send_local_indices(commGrid, index_35);
    MPI_Barrier(MPI_COMM_WORLD);
    double t5_end = MPI_Wtime();
    double t5 = t5_end - t5_start;

    double t6_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    local_join(commGrid, index_05, index_35, 2, 2, 1, 1, order1, index_035_0);
    MPI_Barrier(MPI_COMM_WORLD);
    double t6_end = MPI_Wtime();
    double t6 = t6_end - t6_start;

    send_local_results(commGrid, index_035_0.size() / 3);

    double t7_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    local_redistribution(m_43, index_035_0, 3, 1, index_035);
    MPI_Barrier(MPI_COMM_WORLD);
    double t7_end = MPI_Wtime();
    index_035_0.clear();
    double t7 = t7_end - t7_start;

    send_local_results(commGrid, index_035.size() / 3);

    double t8_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    get_local_indices(m_43, index_43);
    MPI_Barrier(MPI_COMM_WORLD);
    double t8_end = MPI_Wtime();
    double t8 = t8_end - t8_start;

    double t9_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    send_local_indices(commGrid, index_43);
    MPI_Barrier(MPI_COMM_WORLD);
    double t9_end = MPI_Wtime();
    double t9 = t9_end - t9_start;

    double t10_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    local_join(commGrid, index_035, index_43, 3, 2, 1, 1, order2, index_0345_0);
    MPI_Barrier(MPI_COMM_WORLD);
    double t10_end = MPI_Wtime();
    double t10 = t10_end - t10_start;

//    send_local_results(commGrid, index_0345_0.size() / 4);

    double t11_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    local_redistribution(m_64, index_0345_0, 4, 2, index_0345);
    MPI_Barrier(MPI_COMM_WORLD);
    double t11_end = MPI_Wtime();
    index_0345_0.clear();
    double t11 = t11_end - t11_start;

//    send_local_results(commGrid, index_0345.size() / 4);

    double t12_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    get_local_indices(m_64, index_64);
    MPI_Barrier(MPI_COMM_WORLD);
    double t12_end = MPI_Wtime();
    double t12 = t12_end - t12_start;

    double t13_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    send_local_indices(commGrid, index_64);
    MPI_Barrier(MPI_COMM_WORLD);
    double t13_end = MPI_Wtime();
    double t13 = t13_end - t13_start;

    double t14_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    local_filter(commGrid, index_0345, index_64, 4, 2, 2, 3, 1, 0, order3, index_03456);
    MPI_Barrier(MPI_COMM_WORLD);
    double t14_end = MPI_Wtime();
    double t14 = t14_end - t14_start;

//    send_local_results(commGrid, index_03456.size() / 4);

    double t15_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    get_local_indices(m_24, index_24);
    MPI_Barrier(MPI_COMM_WORLD);
    double t15_end = MPI_Wtime();
    double t15 = t15_end - t15_start;

    double t16_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    send_local_indices(commGrid, index_24);
    MPI_Barrier(MPI_COMM_WORLD);
    double t16_end = MPI_Wtime();
    double t16 = t16_end - t16_start;

    double t17_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    local_join(commGrid, index_03456, index_24, 4, 2, 2, 1, order4, index_023456_0);
    MPI_Barrier(MPI_COMM_WORLD);
    double t17_end = MPI_Wtime();
    double t17 = t17_end - t17_start;

//    send_local_results(commGrid, index_023456_0.size() / 5);

    double t18_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    local_redistribution(m_13, index_023456_0, 5, 2, index_023456);
    MPI_Barrier(MPI_COMM_WORLD);
    double t18_end = MPI_Wtime();
    index_023456_0.clear();
    double t18 = t18_end - t18_start;

//    send_local_results(commGrid, index_023456.size() / 5);

    double t19_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    get_local_indices(m_13, index_13);
    MPI_Barrier(MPI_COMM_WORLD);
    double t19_end = MPI_Wtime();
    double t19 = t19_end - t19_start;

    double t20_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    send_local_indices(commGrid, index_13);
    MPI_Barrier(MPI_COMM_WORLD);
    double t20_end = MPI_Wtime();
    double t20 = t20_end - t20_start;

    double t21_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    local_join(commGrid, index_023456, index_13, 5, 2, 2, 1, order5, index_0123456);
    MPI_Barrier(MPI_COMM_WORLD);
    double t21_end = MPI_Wtime();
    double t21 = t21_end - t21_start;

    double t22_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    send_local_results(commGrid, index_0123456.size() / 6);
    MPI_Barrier(MPI_COMM_WORLD);
    double t22_end = MPI_Wtime();
    double t22 = t22_end - t22_start;

    if (myrank == 0) {
        cout << "\tm_05 get local indices takes : " << t2 << " s" << endl;
        cout << "\tm_05 send local indices takes : " << t3 << " s" << endl;
        cout << "\tm_35 get local indices takes : " << t4 << " s" << endl;
        cout << "\tm_35 send local indices takes : " << t5 << " s" << endl;
        cout << "\tm_43 get local indices takes : " << t8 << " s" << endl;
        cout << "\tm_43 send local indices takes : " << t9 << " s" << endl;
        cout << "\tm_64 get local indices takes : " << t12 << " s" << endl;
        cout << "\tm_64 send local indices takes : " << t13 << " s" << endl;
        cout << "\tm_24 get local indices takes : " << t15 << " s" << endl;
        cout << "\tm_24 send local indices takes : " << t16 << " s" << endl;
        cout << "\tm_13 get local indices takes : " << t19 << " s" << endl;
        cout << "\tm_13 send local indices takes : " << t20 << " s" << endl;
        cout << "total get local indices time : " << (t2 + t4 + t8 + t12 + t15 + t19) << " s" << endl;
        cout << "total send local indices time : " << (t3 + t5 + t9 + t13 + t16 + t20) << " s\n" << endl;

        cout << "\tlocal join of m_05 and m_35 takes : " << t6 << " s" << endl;
        cout << "\tlocal join of m_035 and m_43 takes : " << t10 << " s" << endl;
        cout << "\tlocal join of m_03456 and m_24 takes : " << t17 << " s" << endl;
        cout << "\tlocal join of m_023456 and m_13 takes : " << t21 << " s" << endl;
        cout << "total local join time : " << (t6 + t10 + t17 + t21) << " s\n" << endl;

        cout << "\tlocal filter of m_0345 and m_64 takes : " << t14 << " s" << endl;
        cout << "total local filter time : " << t14 << " s\n" << endl;

        cout << "\tredistribution of m_035 takes : " << t7 << " s" << endl;
        cout << "\tredistribution of m_0345 takes : " << t11 << " s" << endl;
        cout << "\tredistribution of m_023456 takes : " << t18 << " s" << endl;
        cout << "total redistribution time : " << (t7 + t11 + t18) << " s" << endl;

        cout << "total send local result time : " << t22 << " s" << endl;
        cout << "---------------------------------------------------------------\n" << endl;
    }
}

void lubm10240_l1(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG, FullyDistVec<IndexType, IndexType> &nonisov) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    total_reduce_time = 0.0;
    total_prune_time = 0.0;
//    total_construct_diag_time = 0.0;
    total_mmul_scalar_time = 0.0;
    total_dim_apply_time = 0.0;

    auto commWorld = G.getcommgrid();

    FullyDistVec<IndexType, ElementType> dm_50(commWorld), dm_35(commWorld), dm_13(commWorld),
            dm_43(commWorld), dm_24(commWorld), dm_35_1(commWorld), dm_64(commWorld), dm_64_1(commWorld),
            dm_43_1(commWorld), dm_35_2(commWorld);

    auto m_50(G), m_35(G), m_43(tG), m_24(tG), m_64(G), m_13(tG);

    IndexType ind1 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(22638)))[0];
    IndexType ind2 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(24)))[0];
    IndexType ind3 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(8622222)))[0];

    FullyDistVec<IndexType, ElementType> r_50(commWorld, G.getnrow(), 0), l_13(commWorld, G.getnrow(), 0), l_24(
            commWorld, G.getnrow(), 0);
    r_50.SetElement(ind1, 6);
    l_13.SetElement(ind2, 1);
    l_24.SetElement(ind3, 1);

    // start count time
    double total_computing_1 = MPI_Wtime();

    // ==> step 1
    if (myrank == 0) {
        cout << "\n###############################################################" << endl;
        cout << "Query 1" << endl;
        cout << "###############################################################" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "step 1 : m_(5,0) = G x {1@(1345,1345)}*6" << endl;
    }
    double t1_start = MPI_Wtime();
    multDimApplyPrune(m_50, r_50, Column, true);
    double t1_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 1 (Total) : " << (t1_end - t1_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 2
    if (myrank == 0) {
        cout << "step 2 : m_(3, 5) = G * m_(5, 0).D()*13" << endl;
    }
    double t2_start = MPI_Wtime();
    diagonalizeV(m_50, dm_50, Row, 10);
    multDimApplyPrune(m_35, dm_50, Column, true);
    double t2_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 2 (Total) : " << (t2_end - t2_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 3
    if (myrank == 0) {
        cout << "step 3 : m_(1,3) = G.T() x m_(3,5).D()*6" << endl;
    }
    double t3_start = MPI_Wtime();
    diagonalizeV(m_35, dm_35, Row, 6);
    multDimApplyPrune(m_13, dm_35, Column, true);
    double t3_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 3 (Total) : " << (t3_end - t3_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 4
    if (myrank == 0) {
        cout << "step 4 : m_(1,3) = {1@(43,43)} x m_(1,3)" << endl;
    }
    double t4_start = MPI_Wtime();
    multDimApplyPrune(m_13, l_13, Row, false);
    double t4_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 4 (Total) : " << (t4_end - t4_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 5
    if (myrank == 0) {
        cout << "step 5 : m_(4,3) = G.T() x m_(1,3).T().D()*8" << endl;
    }
    double t5_start = MPI_Wtime();
    diagonalizeV(m_13, dm_13, Column, 2);
    multDimApplyPrune(m_43, dm_13, Column, true);
    double t5_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 5 (Total) : " << (t5_end - t5_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 6
    if (myrank == 0) {
        cout << "step 6 : m_(2,4) = G.T() x m_(4,3).D()*6" << endl;
    }
    double t6_start = MPI_Wtime();
    diagonalizeV(m_43, dm_43, Row, 6);
    multDimApplyPrune(m_24, dm_43, Column, true);
    double t6_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 6 (Total) : " << (t6_end - t6_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 7
    if (myrank == 0) {
        cout << "step 7 : m_(2,4) = {1@(79,79)} x m_(2,4)" << endl;
    }
    double t7_start = MPI_Wtime();
    multDimApplyPrune(m_24, l_24, Row, false);
    double t7_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 7 (Total) : " << (t7_end - t7_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 8
    if (myrank == 0) {
        cout << "step 8 : m_(6,4) = G x m_(2,4).T().D()*4" << endl;
    }
    double t8_start = MPI_Wtime();
    diagonalizeV(m_24, dm_24, Column, 11);
    multDimApplyPrune(m_64, dm_24, Column, true);
    double t8_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 8 (Total) : " << (t8_end - t8_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 9
    if (myrank == 0) {
        cout << "step 9 : m_(6,4) = m_(3,5).T().D() x m_(6,4)" << endl;
    }
    double t9_start = MPI_Wtime();
    diagonalizeV(m_35, dm_35_1, Column);
    multDimApplyPrune(m_64, dm_35_1, Row, false);
    double t9_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 9 (Total) : " << (t9_end - t9_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 10
    if (myrank == 0) {
        cout << "step 10 : m_(3,5) = m_(3,5) x m_(6,4).D()" << endl;
    }
    double t10_start = MPI_Wtime();
    diagonalizeV(m_64, dm_64);
    multDimApplyPrune(m_35, dm_64, Column, false);
    double t10_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 10 (Total) : " << (t10_end - t10_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 11
    if (myrank == 0) {
        cout << "step 11 : m_(4,3) = m_(6,4).T().D() x m_(4,3)" << endl;
    }
    double t11_start = MPI_Wtime();
    diagonalizeV(m_64, dm_64_1, Column);
    multDimApplyPrune(m_43, dm_64_1, Row, false);
    double t11_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 11 (Total) : " << (t11_end - t11_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 12
    if (myrank == 0) {
        cout << "step 12 : m_(3,5) = m_(4,3).T().D() x m_(3,5)" << endl;
    }
    double t12_start = MPI_Wtime();
    diagonalizeV(m_43, dm_43_1, Column);
    multDimApplyPrune(m_35, dm_43_1, Row, false);
    double t12_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 12 (Total) : " << (t12_end - t12_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 13
    if (myrank == 0) {
        cout << "step 13 : m_(5,0) = m_(3,5).T().D() x m_(5,0)" << endl;
    }
    double t13_start = MPI_Wtime();
    diagonalizeV(m_35, dm_35_2, Column);
    multDimApplyPrune(m_50, dm_35_2, Row, false);
    double t13_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 13 (Total) : " << (t13_end - t13_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    double resgen_start = MPI_Wtime();
    resgen_l1(m_50, m_35, m_43, m_64, m_24, m_13);
    double resgen_end = MPI_Wtime();

    // end count time
    double total_computing_2 = MPI_Wtime();

//    printReducedInfo(m_50);

    if (myrank == 0) {
        cout << "query1 mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "query1 prune time : " << total_prune_time << " s" << endl;
        cout << "query1 diag_reduce time : " << total_reduce_time << " s" << endl;
        cout << "query1 dim_apply time : " << total_dim_apply_time << " s" << endl;
        cout << "query1 result_enum time : " << resgen_end - resgen_start << " s" << endl;
        cout << "query1 time (Total) : " << total_computing_2 - total_computing_1 << " s" << endl;
    }


}

void resgen_l2(PSpMat::MPI_DCCols &m_10, PSpMat::MPI_DCCols &m_21) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    total_get_local_indices_time = 0.0;
    total_send_local_indices_time = 0.0;
    total_local_join_time = 0.0;
    total_local_filter_time = 0.0;
    total_redistribution_time = 0.0;
    total_send_result_time = 0.0;

    if (myrank == 0) {
//        cout << "---------------------------------------------------------------" << endl;
        cout << "begin result generation ......" << endl;
    }

    auto commGrid = m_10.getcommgrid();

    // m_10 becoms m_01
    m_10.Transpose();
    vector<IndexType> index_01;
    get_local_indices(m_10, index_01);
    send_local_indices(commGrid, index_01);
//    write_local_vector(index_01, "m_10", 2);

    vector<IndexType> index_21;
    get_local_indices(m_21, index_21);
    send_local_indices(commGrid, index_21);
//    write_local_vector(index_21, "m_12", 2);

    vector<IndexType> order1 = {0, 0, 0, 1, 1, 0};
    vector<IndexType> index_012;
    local_join(commGrid, index_01, index_21, 2, 2, 1, 1, order1, index_012);
//    write_local_vector(index_012, "index_012", 3);

    send_local_results(commGrid, index_012.size() / 3);
}

void lubm10240_l2(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG, FullyDistVec<IndexType, IndexType> &nonisov) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    total_reduce_time = 0.0;
    total_prune_time = 0.0;
//    total_construct_diag_time = 0.0;
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
    double resgen_start = MPI_Wtime();
    resgen_l2(m_10, m_21);
    double resgen_end = MPI_Wtime();

    // end count time
    double total_computing_2 = MPI_Wtime();

//    printReducedInfo(m_10);

    if (myrank == 0) {
        cout << "query2 mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "query2 prune time : " << total_prune_time << " s" << endl;
        cout << "query2 diag_reduce time : " << total_reduce_time << " s" << endl;
        cout << "query2 dim_apply time : " << total_dim_apply_time << " s" << endl;
        cout << "query2 result_enum time : " << resgen_end - resgen_start << " s" << endl;
        cout << "query2 time (Total) : " << total_computing_2 - total_computing_1 << " s" << endl;
    }
}

void lubm10240_l3(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG, FullyDistVec<IndexType, IndexType> &nonisov) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    total_reduce_time = 0.0;
    total_prune_time = 0.0;
//    total_construct_diag_time = 0.0;
    total_mmul_scalar_time = 0.0;
    total_dim_apply_time = 0.0;

    auto commWorld = G.getcommgrid();

    FullyDistVec<IndexType, ElementType> dm_50(commWorld), dm_35(commWorld), dm_13(commWorld),
            dm_43(commWorld), dm_24(commWorld), dm_35_1(commWorld), dm_64(commWorld), dm_64_1(commWorld),
            dm_43_1(commWorld), dm_35_2(commWorld);

    auto m_50(G), m_35(G), m_13(tG), m_43(tG), m_24(tG), m_64(G);

    IndexType ind1 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(22638)))[0];
    IndexType ind2 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(43)))[0];
    IndexType ind3 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(8622222)))[0];

    FullyDistVec<IndexType, ElementType> r_50(commWorld, G.getnrow(), 0), l_13(commWorld, G.getnrow(), 0), l_24(
            commWorld, G.getnrow(), 0);
    r_50.SetElement(ind1, 6);
    l_13.SetElement(ind2, 1);
    l_24.SetElement(ind3, 1);

    // start count time
    double total_computing_1 = MPI_Wtime();

    // ==> step 1
    if (myrank == 0) {
        cout << "\n###############################################################" << endl;
        cout << "Query 3" << endl;
        cout << "###############################################################" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "step 1 : m_(5,0) = G x {1@(22638,22638)}*6" << endl;
    }
    double t1_start = MPI_Wtime();
    multDimApplyPrune(m_50, r_50, Column, true);
    double t1_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 1 (Total) : " << (t1_end - t1_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 2
    if (myrank == 0) {
        cout << "step 2 : m_(3, 5) = G * m_(5, 0).D()*10" << endl;
    }
    double t2_start = MPI_Wtime();
    diagonalizeV(m_50, dm_50, Row, 10);
    multDimApplyPrune(m_35, dm_50, Column, true);
    double t2_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 2 (Total) : " << (t2_end - t2_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 3
    if (myrank == 0) {
        cout << "step 3 : m_(1,3) = G.T() x m_(3,5).D()*6" << endl;
    }
    double t3_start = MPI_Wtime();
    diagonalizeV(m_35, dm_35, Row, 6);
    multDimApplyPrune(m_13, dm_35, Column, true);
    double t3_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 3 (Total) : " << (t3_end - t3_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 4
    if (myrank == 0) {
        cout << "step 4 : m_(1,3) = {1@(43,43)} x m_(1,3)" << endl;
    }
    double t4_start = MPI_Wtime();
    multDimApplyPrune(m_13, l_13, Row, false);
    double t4_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 4 (Total) : " << (t4_end - t4_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 5
    if (myrank == 0) {
        cout << "step 5 : m_(4,3) = G.T() x m_(1,3).T().D()*2" << endl;
    }
    double t5_start = MPI_Wtime();
    diagonalizeV(m_13, dm_13, Column, 2);
    multDimApplyPrune(m_43, dm_13, Column, true);
    double t5_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 5 (Total) : " << (t5_end - t5_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

//    // ==> step 6
//    if (myrank == 0) {
//        cout << "step 6 : m_(2,4) = G.T() x m_(4,3).D()*6" << endl;
//    }
//    double t6_start = MPI_Wtime();
//    diagonalizeV(m_43, dm_43, Row, 6);
//    multDimApplyPrune(m_24, dm_43, Column, true);
//    double t6_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 6 (Total) : " << (t6_end - t6_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }
//
//    // ==> step 7
//    if (myrank == 0) {
//        cout << "step 7 : m_(2,4) = {1@(862222,862222)} x m_(2,4)" << endl;
//    }
//    double t7_start = MPI_Wtime();
//    multDimApplyPrune(m_24, l_24, Row, false);
//    double t7_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 7 (Total) : " << (t7_end - t7_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }
//
//    // ==> step 8
//    if (myrank == 0) {
//        cout << "step 8 : m_(6,4) = G x m_(2,4).T().D()*11" << endl;
//    }
//    double t8_start = MPI_Wtime();
//    diagonalizeV(m_24, dm_24, Column, 11);
//    multDimApplyPrune(m_64, dm_24, Column, true);
//    double t8_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 8 (Total) : " << (t8_end - t8_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }
//
//    // ==> step 9
//    if (myrank == 0) {
//        cout << "step 9 : m_(6,4) = m_(3,5).T().D() x m_(6,4)" << endl;
//    }
//    double t9_start = MPI_Wtime();
//    diagonalizeV(m_35, dm_35_1, Column);
//    multDimApplyPrune(m_64, dm_35_1, Row, false);
//    double t9_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 9 (Total) : " << (t9_end - t9_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }
//
//    // ==> step 10
//    if (myrank == 0) {
//        cout << "step 10 : m_(3,5) = m_(3,5) x m_(6,4).D()" << endl;
//    }
//    double t10_start = MPI_Wtime();
//    diagonalizeV(m_64, dm_64);
//    multDimApplyPrune(m_35, dm_64, Column, false);
//    double t10_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 10 (Total) : " << (t10_end - t10_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }
//
//    // ==> step 11
//    if (myrank == 0) {
//        cout << "step 11 : m_(4,3) = m_(6,4).T().D() x m_(4,3)" << endl;
//    }
//    double t11_start = MPI_Wtime();
//    diagonalizeV(m_64, dm_64_1, Column);
//    multDimApplyPrune(m_43, dm_64_1, Row, false);
//    double t11_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 11 (Total) : " << (t11_end - t11_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }
//
//    // ==> step 12
//    if (myrank == 0) {
//        cout << "step 12 : m_(3,5) = m_(4,3).T().D() x m_(3,5)" << endl;
//    }
//    double t12_start = MPI_Wtime();
//    diagonalizeV(m_43, dm_43_1, Column);
//    multDimApplyPrune(m_35, dm_43_1, Row, false);
//    double t12_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 12 (Total) : " << (t12_end - t12_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }
//
//    // ==> step 13
//    if (myrank == 0) {
//        cout << "step 13 : m_(5,0) = m_(3,5).T().D() x m_(5,0)" << endl;
//    }
//    double t13_start = MPI_Wtime();
//    diagonalizeV(m_35, dm_35_2, Column);
//    multDimApplyPrune(m_50, dm_35_2, Row, false);
//    double t13_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 13 (Total) : " << (t13_end - t13_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }

    double resgen_start = MPI_Wtime();
    double resgen_end = MPI_Wtime();


    // end count time
    double total_computing_2 = MPI_Wtime();

//    printReducedInfo(m_43);

    if (myrank == 0) {
        cout << "query3 mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "query3 prune time : " << total_prune_time << " s" << endl;
        cout << "query3 diag_reduce time : " << total_reduce_time << " s" << endl;
        cout << "query3 dim_apply time : " << total_dim_apply_time << " s" << endl;
        cout << "query3 result_enum time : " << resgen_end - resgen_start << " s" << endl;
        cout << "query3 time (Total) : " << total_computing_2 - total_computing_1 << " s" << endl;
    }

    // TODO : no need to do result generation because of empty result
}

// TODO : smaller difference with other implementation results
void resgen_l4(PSpMat::MPI_DCCols &m_20, PSpMat::MPI_DCCols &m_52, PSpMat::MPI_DCCols &m_42, PSpMat::MPI_DCCols &m_32,
               PSpMat::MPI_DCCols &m_12) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    total_get_local_indices_time = 0.0;
    total_send_local_indices_time = 0.0;
    total_local_join_time = 0.0;
    total_local_filter_time = 0.0;
    total_redistribution_time = 0.0;
    total_send_result_time = 0.0;

    if (myrank == 0) {
//        cout << "---------------------------------------------------------------" << endl;
        cout << "begin result generation ......" << endl;
    }

    auto commGrid = m_20.getcommgrid();

    // m_20 becoms m_02
    m_20.Transpose();
    vector<IndexType> index_02;
    get_local_indices(m_20, index_02);
    send_local_indices(commGrid, index_02);

    vector<IndexType> index_52;
    get_local_indices(m_52, index_52);
    send_local_indices(commGrid, index_52);

    vector<IndexType> order1 = {0, 0, 0, 1, 1, 0};
    vector<IndexType> index_025;
    local_join(commGrid, index_02, index_52, 2, 2, 1, 1, order1, index_025);

    vector<IndexType> index_42;
    get_local_indices(m_42, index_42);
    send_local_indices(commGrid, index_42);

    vector<IndexType> order2 = {0, 0, 0, 1, 1, 0, 0, 2};
    vector<IndexType> index_0245;
    local_join(commGrid, index_025, index_42, 3, 2, 1, 1, order2, index_0245);

    vector<IndexType> index_32;
    get_local_indices(m_32, index_32);
    send_local_indices(commGrid, index_32);

    vector<IndexType> order3 = {0, 0, 0, 1, 1, 0, 0, 2, 0, 3};
    vector<IndexType> index_02345;
    local_join(commGrid, index_0245, index_32, 4, 2, 1, 1, order3, index_02345);

    vector<IndexType> index_12;
    get_local_indices(m_12, index_12);
    send_local_indices(commGrid, index_12);

    vector<IndexType> order4 = {0, 0, 1, 0, 0, 1, 0, 2, 0, 3, 0, 4};
    vector<IndexType> index_012345;
    local_join(commGrid, index_02345, index_12, 5, 2, 1, 1, order2, index_012345);

    send_local_results(commGrid, index_012345.size() / 6);
}

void lubm10240_l4(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG, FullyDistVec<IndexType, IndexType> &nonisov) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    total_reduce_time = 0.0;
    total_prune_time = 0.0;
//    total_construct_diag_time = 0.0;
    total_mmul_scalar_time = 0.0;
    total_dim_apply_time = 0.0;

    auto commWorld = G.getcommgrid();

    FullyDistVec<IndexType, ElementType> dm_20(commWorld), dm_12(commWorld), dm_32(commWorld), dm_42(commWorld), dm_52(
            commWorld);

    auto m_20(G), m_12(tG), m_32(tG), m_42(tG), m_52(tG);

    IndexType ind1 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(11)))[0];
    IndexType ind2 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(1345)))[0];

    FullyDistVec<IndexType, ElementType> r_20(commWorld, G.getnrow(), 0), l_12(commWorld, G.getnrow(), 0);
    r_20.SetElement(ind1, 5);
    l_12.SetElement(ind2, 1);

    // start count time
    double total_computing_1 = MPI_Wtime();

    // ==> step 1
    if (myrank == 0) {
        cout << "\n###############################################################" << endl;
        cout << "Query 4" << endl;
        cout << "###############################################################" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "step 1 : m_(2,0) = G x {1@(11,11)}*5" << endl;
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
        cout << "step 2 : m_(1,2) = G.T() x m_(2,0).D()*6" << endl;
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
        cout << "step 3 : m_(1,0) = m_(2,1).T().D() x m_(1,0)" << endl;
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
        cout << "step 4 : m_(3,2) = G.T() x m_(1,2).T().D()*3" << endl;
    }
    double t4_start = MPI_Wtime();
    diagonalizeV(m_12, dm_12, Column, 3);
    multDimApplyPrune(m_32, dm_12, Column, true);
    double t4_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 4 (Total) : " << (t4_end - t4_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 5
    if (myrank == 0) {
        cout << "step 5 : m_(4,2) = G.T() x m_(3,2).T().D()*12" << endl;
    }
    double t5_start = MPI_Wtime();
    diagonalizeV(m_32, dm_32, Column, 12);
    multDimApplyPrune(m_42, dm_32, Column, true);
    double t5_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 5 (Total) : " << (t5_end - t5_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 6
    if (myrank == 0) {
        cout << "step 6 : m_(5,2) = G.T() x m_(4,2).T().D()*9" << endl;
    }
    double t6_start = MPI_Wtime();
    diagonalizeV(m_42, dm_42, Column, 9);
    multDimApplyPrune(m_52, dm_42, Column, true);
    double t6_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 6 (Total) : " << (t6_end - t6_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 7
    if (myrank == 0) {
        cout << "step 7 : m_(2,0) = m_(5,2).T().D() x m_(2,0)" << endl;
    }
    double t7_start = MPI_Wtime();
    diagonalizeV(m_52, dm_52, Column);
    multDimApplyPrune(m_20, dm_52, Row, true);
    double t7_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 7 (Total) : " << (t7_end - t7_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    double resgen_start = MPI_Wtime();
    resgen_l4(m_20, m_52, m_42, m_32, m_12);
    double resgen_end = MPI_Wtime();

    // end count time
    double total_computing_2 = MPI_Wtime();

//    printReducedInfo(m_20);

    if (myrank == 0) {
        cout << "query4 mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "query4 prune time : " << total_prune_time << " s" << endl;
        cout << "query4 diag_reduce time : " << total_reduce_time << " s" << endl;
        cout << "query4 dim_apply time : " << total_dim_apply_time << " s" << endl;
        cout << "query4 result_enum time : " << resgen_end - resgen_start << " s" << endl;
        cout << "query4 time (Total) : " << total_computing_2 - total_computing_1 << " s" << endl;
    }

}

void resgen_l5(PSpMat::MPI_DCCols &m_20, PSpMat::MPI_DCCols &m_12) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    total_get_local_indices_time = 0.0;
    total_send_local_indices_time = 0.0;
    total_local_join_time = 0.0;
    total_local_filter_time = 0.0;
    total_redistribution_time = 0.0;
    total_send_result_time = 0.0;

    if (myrank == 0) {
//        cout << "---------------------------------------------------------------" << endl;
        cout << "begin result generation ......" << endl;
    }

    auto commGrid = m_20.getcommgrid();

    // m_20 becoms m_02
    m_20.Transpose();
    vector<IndexType> index_02;
    get_local_indices(m_20, index_02);
    send_local_indices(commGrid, index_02);
//    write_local_vector(index_02, "m_20", 2);

    vector<IndexType> index_12;
    get_local_indices(m_12, index_12);
    send_local_indices(commGrid, index_12);
//    write_local_vector(index_12, "m_12", 2);

    vector<IndexType> order1 = {0, 1, 1, 1, 0, 0};
    vector<IndexType> index_012;
    local_join(commGrid, index_02, index_12, 2, 2, 1, 1, order1, index_012);
//    write_local_vector(index_012, "index_012", 3);

    send_local_results(commGrid, index_012.size() / 3);

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

    double resgen_start = MPI_Wtime();
    resgen_l5(m_20, m_12);
    double resgen_end = MPI_Wtime();

    // end count time
    double total_computing_2 = MPI_Wtime();

//    printReducedInfo(m_20);

    if (myrank == 0) {
        cout << "query5 mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "query5 prune time : " << total_prune_time << " s" << endl;
        cout << "query5 diag_reduce time : " << total_reduce_time << " s" << endl;
        cout << "query5 dim_apply time : " << total_dim_apply_time << " s" << endl;
        cout << "query5 result_enum time : " << resgen_end - resgen_start << " s" << endl;
        cout << "query5 time (Total) : " << total_computing_2 - total_computing_1 << " s" << endl;
    }
}

void resgen_l6(PSpMat::MPI_DCCols &m_30, PSpMat::MPI_DCCols &m_43, PSpMat::MPI_DCCols &m_14, PSpMat::MPI_DCCols &m_24) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    total_get_local_indices_time = 0.0;
    total_send_local_indices_time = 0.0;
    total_local_join_time = 0.0;
    total_local_filter_time = 0.0;
    total_redistribution_time = 0.0;
    total_send_result_time = 0.0;

    if (myrank == 0) {
//        cout << "---------------------------------------------------------------" << endl;
        cout << "begin result generation ......" << endl;
    }

    auto commGrid = m_30.getcommgrid();

    // m_30 becoms m_03
    m_30.Transpose();
    vector<IndexType> index_03;
    get_local_indices(m_30, index_03);
    send_local_indices(commGrid, index_03);
//    write_local_vector(index_03, "m_30", 2);

    vector<IndexType> index_43;
    get_local_indices(m_43, index_43);
    send_local_indices(commGrid, index_43);
//    write_local_vector(index_43, "m_43", 2);

    vector<IndexType> order1 = {0, 0, 0, 1, 2, 0};
    vector<IndexType> index_034_0, index_034;
    local_join(commGrid, index_03, index_43, 2, 2, 1, 1, order1, index_034_0);
//    write_local_vector(index_034_0, "index_034_0", 3);
    local_redistribution(m_14, index_034_0, 3, 2, index_034);

    vector<IndexType> index_24;
    get_local_indices(m_24, index_24);
    send_local_indices(commGrid, index_24);
//    write_local_vector(index_24, "m_24", 2);

    vector<IndexType> order2 = {0, 0, 1, 0, 0, 1, 0, 2};
    vector<IndexType> index_0234;
    local_join(commGrid, index_034, index_24, 3, 2, 2, 1, order2, index_0234);
//    write_local_vector(index_0234, "index_0234", 4);

    vector<IndexType> index_14;
    get_local_indices(m_14, index_14);
    send_local_indices(commGrid, index_14);
//    write_local_vector(index_14, "m_14", 2);

    vector<IndexType> order3 = {0, 0, 1, 0, 0, 1, 0, 2, 0, 3};
    vector<IndexType> index_01234;
    local_join(m_14.getcommgrid(), index_0234, index_14, 4, 2, 3, 1, order3, index_01234);
//    write_local_vector(index_01234, "index_01234", 5);

    send_local_results(commGrid, index_01234.size() / 5);

}

void lubm10240_l6(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG, FullyDistVec<IndexType, IndexType> &nonisov) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    total_reduce_time = 0.0;
    total_prune_time = 0.0;
//    total_construct_diag_time = 0.0;
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

    double resgen_start = MPI_Wtime();
    resgen_l6(m_30, m_43, m_14, m_24);
    double resgen_end = MPI_Wtime();

    // end count time
    double total_computing_2 = MPI_Wtime();

//    printReducedInfo(m_30);

    if (myrank == 0) {
        cout << "query6 mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "query6 prune time : " << total_prune_time << " s" << endl;
        cout << "query6 diag_reduce time : " << total_reduce_time << " s" << endl;
        cout << "query6 dim_apply time : " << total_dim_apply_time << " s" << endl;
        cout << "query6 result_enum time : " << resgen_end - resgen_start << " s" << endl;
        cout << "query6 time (Total) : " << total_computing_2 - total_computing_1 << " s" << endl;
    }
}

void resgen_l7(PSpMat::MPI_DCCols &m_50, PSpMat::MPI_DCCols &m_35, PSpMat::MPI_DCCols &m_43, PSpMat::MPI_DCCols &m_64,
               PSpMat::MPI_DCCols &m_24, PSpMat::MPI_DCCols &m_13) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    total_get_local_indices_time = 0.0;
    total_send_local_indices_time = 0.0;
    total_local_join_time = 0.0;
    total_local_filter_time = 0.0;
    total_redistribution_time = 0.0;
    total_send_result_time = 0.0;

    if (myrank == 0) {
        cout << "begin result generation ......" << endl;
    }

    auto commGrid = m_50.getcommgrid();

    // m_50 becoms m_05
    m_50.Transpose();
    vector<IndexType> index_05;
    get_local_indices(m_50, index_05);
    send_local_indices(commGrid, index_05);

    vector<IndexType> index_35;
    get_local_indices(m_35, index_35);
    send_local_indices(commGrid, index_35);

    vector<IndexType> order1 = {0, 0, 1, 0, 1, 1};
    vector<IndexType> index_035_0, index_035;
    local_join(commGrid, index_05, index_35, 2, 2, 1, 1, order1, index_035_0);
    local_redistribution(m_43, index_035_0, 3, 1, index_035);
    index_035_0.clear();

    vector<IndexType> index_43;
    get_local_indices(m_43, index_43);
    send_local_indices(commGrid, index_43);

    vector<IndexType> order2 = {0, 0, 0, 1, 1, 0, 0, 2};
    vector<IndexType> index_0345_0, index_0345;

    local_join(commGrid, index_035, index_43, 3, 2, 1, 1, order2, index_0345_0);

    local_redistribution(m_64, index_0345_0, 4, 2, index_0345);
    index_0345_0.clear();

    vector<IndexType> index_64;
    get_local_indices(m_64, index_64);
    send_local_indices(commGrid, index_64);

    vector<IndexType> order3 = {0, 0, 0, 1, 0, 2, 0, 3};
    vector<IndexType> index_03456;

    local_filter(commGrid, index_0345, index_64, 4, 2, 2, 3, 1, 0, order3, index_03456);

    vector<IndexType> index_24;
    get_local_indices(m_24, index_24);
    send_local_indices(commGrid, index_24);

    vector<IndexType> order4 = {0, 0, 1, 0, 0, 1, 0, 2, 0, 3};
    vector<IndexType> index_023456_0, index_023456;

    local_join(commGrid, index_03456, index_24, 4, 2, 2, 1, order4, index_023456_0);

    local_redistribution(m_13, index_023456_0, 5, 2, index_023456);
    index_023456_0.clear();

    vector<IndexType> index_13;
    get_local_indices(m_13, index_13);
    send_local_indices(commGrid, index_13);

    vector<IndexType> order5 = {0, 0, 1, 0, 0, 1, 0, 2, 0, 3, 0, 4};
    vector<IndexType> index_0123456;
    local_join(commGrid, index_023456, index_13, 5, 2, 2, 1, order5, index_0123456);

    send_local_results(commGrid, index_0123456.size() / 6);
}

// you have two choices
// choice == 0 : use multPrune
// choice == 1 : use multDimApplyPrune
void lubm10240_l7(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG, FullyDistVec<IndexType, IndexType> &nonisov) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    total_reduce_time = 0.0;
    total_prune_time = 0.0;
//    total_construct_diag_time = 0.0;
    total_mmul_scalar_time = 0.0;
    total_dim_apply_time = 0.0;

    auto commWorld = G.getcommgrid();

    IndexType ind1 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(1345)))[0];
    IndexType ind2 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(43)))[0];
    IndexType ind3 = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(79)))[0];


    FullyDistVec<IndexType, ElementType> dm_50(commWorld), dm_35(commWorld), dm_13(commWorld),
            dm_43(commWorld), dm_24(commWorld), dm_35_1(commWorld), dm_64(commWorld), dm_64_1(commWorld),
            dm_43_1(commWorld), dm_35_2(commWorld);

    auto m_50(G), m_35(G), m_13(tG), m_43(tG), m_24(tG), m_64(G);

    FullyDistVec<IndexType, ElementType> r_50(commWorld, G.getnrow(), 0), l_13(commWorld, G.getnrow(), 0), l_24(
            commWorld, G.getnrow(), 0);
    r_50.SetElement(ind1, 6);
    l_13.SetElement(ind2, 1);
    l_24.SetElement(ind3, 1);

    // start count time
    double total_computing_1 = MPI_Wtime();

    // ==> step 1
    if (myrank == 0) {
        cout << "\n###############################################################" << endl;
        cout << "Query 7" << endl;
        cout << "###############################################################" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "step 1 : m_(5,0) = G x {1@(1345,1345)}*6" << endl;
    }
    double t1_start = MPI_Wtime();
    multDimApplyPrune(m_50, r_50, Column, true);
    double t1_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 1 (Total) : " << (t1_end - t1_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 2
    if (myrank == 0) {
        cout << "step 2 : m_(3, 5) = G x m_(5, 0).D()*13" << endl;
    }
    double t2_start = MPI_Wtime();
    diagonalizeV(m_50, dm_50, Row, 13);
    multDimApplyPrune(m_35, dm_50, Column, true);
    double t2_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 2 (Total) : " << (t2_end - t2_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 3
    if (myrank == 0) {
        cout << "step 3 : m_(1,3) = G.T() x m_(3,5).D()*6" << endl;
    }
    double t3_start = MPI_Wtime();
    diagonalizeV(m_35, dm_35, Row, 6);
    multDimApplyPrune(m_13, dm_35, Column, true);
    double t3_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 3 (Total) : " << (t3_end - t3_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 4
    if (myrank == 0) {
        cout << "step 4 : m_(1,3) = {1@(43,43)} x m_(1,3)" << endl;
    }
    double t4_start = MPI_Wtime();
    multDimApplyPrune(m_13, l_13, Row, false);
    double t4_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 4 (Total) : " << (t4_end - t4_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 5
    if (myrank == 0) {
        cout << "step 5 : m_(4,3) = G.T() x m_(1,3).T().D()*8" << endl;
    }
    double t5_start = MPI_Wtime();
    diagonalizeV(m_13, dm_13, Column, 8);
    multDimApplyPrune(m_43, dm_13, Column, true);
    double t5_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 5 (Total) : " << (t5_end - t5_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 6
    if (myrank == 0) {
        cout << "step 6 : m_(2,4) = G.T() x m_(4,3).D()*6" << endl;
    }
    double t6_start = MPI_Wtime();
    diagonalizeV(m_43, dm_43, Row, 6);
    multDimApplyPrune(m_24, dm_43, Column, true);
    double t6_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 6 (Total) : " << (t6_end - t6_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 7
    if (myrank == 0) {
        cout << "step 7 : m_(2,4) = {1@(79,79)} x m_(2,4)" << endl;
    }
    double t7_start = MPI_Wtime();
    multDimApplyPrune(m_24, l_24, Row, false);
    double t7_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 7 (Total) : " << (t7_end - t7_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 8
    if (myrank == 0) {
        cout << "step 8 : m_(6,4) = G x m_(2,4).T().D()*4" << endl;
    }
    double t8_start = MPI_Wtime();
    diagonalizeV(m_24, dm_24, Column, 4);
    multDimApplyPrune(m_64, dm_24, Column, true);
    double t8_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 8 (Total) : " << (t8_end - t8_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 9
    if (myrank == 0) {
        cout << "step 9 : m_(6,4) = m_(3,5).T().D() x m_(6,4)" << endl;
    }
    double t9_start = MPI_Wtime();
    diagonalizeV(m_35, dm_35_1, Column);
    multDimApplyPrune(m_64, dm_35_1, Row, false);
    double t9_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 9 (Total) : " << (t9_end - t9_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 10
    if (myrank == 0) {
        cout << "step 10 : m_(3,5) = m_(3,5) x m_(6,4).D()" << endl;
    }
    double t10_start = MPI_Wtime();
    diagonalizeV(m_64, dm_64);
    multDimApplyPrune(m_35, dm_64, Column, false);
    double t10_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 10 (Total) : " << (t10_end - t10_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 11
    if (myrank == 0) {
        cout << "step 11 : m_(4,3) = m_(6,4).T().D() x m_(4,3)" << endl;
    }
    double t11_start = MPI_Wtime();
    diagonalizeV(m_64, dm_64_1, Column);
    multDimApplyPrune(m_43, dm_64_1, Row, false);
    double t11_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 11 (Total) : " << (t11_end - t11_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 12
    if (myrank == 0) {
        cout << "step 12 : m_(3,5) = m_(4,3).T().D() x m_(3,5)" << endl;
    }
    double t12_start = MPI_Wtime();
    diagonalizeV(m_43, dm_43_1, Column);
    multDimApplyPrune(m_35, dm_43_1, Row, false);
    double t12_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 12 (Total) : " << (t12_end - t12_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 13
    if (myrank == 0) {
        cout << "step 13 : m_(5,0) = m_(3,5).T().D() x m_(5,0)" << endl;
    }
    double t13_start = MPI_Wtime();
    diagonalizeV(m_35, dm_35_2, Column);
    multDimApplyPrune(m_50, dm_35_2, Row, false);
    double t13_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 13 (Total) : " << (t13_end - t13_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    double resgen_start = MPI_Wtime();
    resgen_l7(m_50, m_35, m_43, m_64, m_24, m_13);
    double resgen_end = MPI_Wtime();

    // end count time
    double total_computing_2 = MPI_Wtime();

//    printReducedInfo(m_50);

    if (myrank == 0) {
        cout << "query7 mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "query7 prune time : " << total_prune_time << " s" << endl;
        cout << "query7 diag_reduce time : " << total_reduce_time << " s" << endl;
        cout << "query7 dim_apply time : " << total_dim_apply_time << " s" << endl;
        cout << "query7 result_enum time : " << resgen_end - resgen_start << " s" << endl;
        cout << "query7 time (Total) : " << total_computing_2 - total_computing_1 << " s" << endl;
    }
}

int main(int argc, char *argv[]) {
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    comp[0] = compInt3A;
    comp[1] = compInt3B;
    comp[2] = compInt3C;

    comp[5] = compInt4A;
    comp[6] = compInt4B;
    comp[7] = compInt4C;
    comp[8] = compInt4D;

    comp[10] = compInt5A;
    comp[11] = compInt5B;
    comp[12] = compInt5C;
    comp[13] = compInt5D;
    comp[14] = compInt5E;

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

        // query
//        for (int time = 1; time <= 5; time++) {
            lubm10240_l1(G, tG, nonisov);
//            lubm10240_l2(G, tG, nonisov);
//            lubm10240_l3(G, tG, nonisov);
//            lubm10240_l4(G, tG, nonisov);
//            lubm10240_l5(G, tG, nonisov);
//            lubm10240_l6(G, tG, nonisov);
//            lubm10240_l7(G, tG, nonisov);
//        }
    }

    MPI_Finalize();
    return 0;
}
