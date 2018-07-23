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

void resgen_l1(PSpMat::MPI_DCCols &m_30, PSpMat::MPI_DCCols &m_43, PSpMat::MPI_DCCols &m_24, PSpMat::MPI_DCCols &m_54,
               PSpMat::MPI_DCCols &m_15, PSpMat::MPI_DCCols &m_65) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t1_start = MPI_Wtime();
    clear_result_time();

    auto commGrid = m_30.getcommgrid();

    // m_30 becoms m_03
    m_30.Transpose();

    vector<IndexType> indl, indr, indj;

    vector<IndexType> order1 = {0, 0, 0, 1, 1, 0}, order2 = {0, 0, 0, 1, 0, 2, 1, 0}, order3 = {0, 0, 0, 1, 0, 2, 0, 3},
            order4 = {0, 0, 1, 0, 0, 1, 0, 2, 0, 3}, order5 = {0, 0, 0, 1, 1, 0, 0, 2, 0, 3, 0, 4};

    double t1_end = MPI_Wtime();
    if (myrank == 0) {
        cout << "begin result generation ......" << endl;
        cout << "\ttranspose matrix and declarations take : " << (t1_end - t1_start) << " s\n" << endl;
    }

    get_local_indices(m_30, indl);
    send_local_indices(commGrid, indl);
    m_30.FreeMemory();

    get_local_indices(m_43, indr);
    send_local_indices(commGrid, indr);
    m_43.FreeMemory();

    local_join(commGrid, indl, indr, 2, 2, 1, 1, order1, indj);
    local_redistribution(m_54, indj, 3, 2, indl);

    get_local_indices(m_54, indr);
    send_local_indices(commGrid, indr);
    m_54.FreeMemory();

    local_join(commGrid, indl, indr, 3, 2, 2, 1, order2, indj);
    local_redistribution(m_65, indj, 4, 3, indl);

    get_local_indices(m_65, indr);
    send_local_indices(commGrid, indr);
    m_65.FreeMemory();

    local_filter(commGrid, indl, indr, 4, 2, 3, 1, 1, 0, order3, indj);
    indl = indj;

    get_local_indices(m_15, indr);
    send_local_indices(commGrid, indr);
    m_15.FreeMemory();

    local_join(commGrid, indl, indr, 4, 2, 3, 1, order4, indj);
    local_redistribution(m_24, indj, 5, 3, indl);

    get_local_indices(m_24, indr);
    send_local_indices(commGrid, indr);
    m_24.FreeMemory();

    local_join(commGrid, indl, indr, 5, 2, 3, 1, order5, indj);

    send_local_results(commGrid, indj.size() / 6);
}

void lubm10240_l1(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    clear_query_time();

    auto commWorld = G.getcommgrid();

    FullyDistVec<IndexType, ElementType> dm(commWorld);

    auto m_30(G), m_43(tG), m_24(tG), m_54(G), m_15(tG), m_65(G);

    FullyDistVec<IndexType, ElementType> r_30(commWorld, G.getnrow(), 0), l_24(commWorld, G.getnrow(), 0), l_15(commWorld, G.getnrow(), 0);
    r_30.SetElement(12797256, 2);
    l_24.SetElement(24807258, 1);
    l_15.SetElement(23321407, 1);

    // start count time
    double total_computing_1 = MPI_Wtime();

    // ==> step 1
    if (myrank == 0) {
        cout << "\n###############################################################" << endl;
        cout << "Query 1" << endl;
        cout << "###############################################################" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "step 1 : m_(3,0) = G x {1@(12797256,12797256)}*2" << endl;
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
        cout << "step 2 : m_(4,3) = G.T() * m_(3, 0).D()*8" << endl;
    }
    double t2_start = MPI_Wtime();
    diagonalizeV(m_30, dm, Row, 8);
    multDimApplyPrune(m_43, dm, Column, true);
    double t2_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 2 (Total) : " << (t2_end - t2_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 3
    if (myrank == 0) {
        cout << "step 3 : m_(2,4) = G.T() x m_(4,3).D()*2" << endl;
    }
    double t3_start = MPI_Wtime();
    diagonalizeV(m_43, dm, Row, 2);
    multDimApplyPrune(m_24, dm, Column, true);
    double t3_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 3 (Total) : " << (t3_end - t3_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 4
    if (myrank == 0) {
        cout << "step 4 : m_(2,4) = {1@(24807258,24807258)} x m_(2,4)" << endl;
    }
    double t4_start = MPI_Wtime();
    multDimApplyPrune(m_24, l_24, Row, false);
    double t4_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 4 (Total) : " << (t4_end - t4_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 5
    if (myrank == 0) {
        cout << "step 5 : m_(5,4) = G x m_(2,4).T().D()*6" << endl;
    }
    double t5_start = MPI_Wtime();
    diagonalizeV(m_24, dm, Column, 6);
    multDimApplyPrune(m_54, dm, Column, true);
    double t5_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 5 (Total) : " << (t5_end - t5_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 6
    if (myrank == 0) {
        cout << "step 6 : m_(1,5) = G.T() x m_(5,4).D()*2" << endl;
    }
    double t6_start = MPI_Wtime();
    diagonalizeV(m_54, dm, Row, 2);
    multDimApplyPrune(m_15, dm, Column, true);
    double t6_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 6 (Total) : " << (t6_end - t6_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 7
    if (myrank == 0) {
        cout << "step 7 : m_(1,5) = {1@(23321407,23321407)} x m_(1,5)" << endl;
    }
    double t7_start = MPI_Wtime();
    multDimApplyPrune(m_15, l_15, Row, false);
    double t7_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 7 (Total) : " << (t7_end - t7_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 8
    if (myrank == 0) {
        cout << "step 8 : m_(6,5) = G x m_(1,5).T().D()*15" << endl;
    }
    double t8_start = MPI_Wtime();
    diagonalizeV(m_15, dm, Column, 15);
    multDimApplyPrune(m_65, dm, Column, true);
    double t8_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 8 (Total) : " << (t8_end - t8_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 9
    if (myrank == 0) {
        cout << "step 9 : m_(6,5) = m_(4,3).T().D() x m_(6,5)" << endl;
    }
    double t9_start = MPI_Wtime();
    diagonalizeV(m_43, dm, Column);
    multDimApplyPrune(m_65, dm, Row, false);
    double t9_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 9 (Total) : " << (t9_end - t9_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 10
    if (myrank == 0) {
        cout << "step 10 : m_(4,3) = m_(4,3) x m_(6,5).D()" << endl;
    }
    double t10_start = MPI_Wtime();
    diagonalizeV(m_65, dm);
    multDimApplyPrune(m_43, dm, Column, false);
    double t10_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 10 (Total) : " << (t10_end - t10_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 11
    if (myrank == 0) {
        cout << "step 11 : m_(5,4) = m_(6,5).T().D() x m_(5,4)" << endl;
    }
    double t11_start = MPI_Wtime();
    diagonalizeV(m_65, dm, Column);
    multDimApplyPrune(m_54, dm, Row, false);
    double t11_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 11 (Total) : " << (t11_end - t11_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 12
    if (myrank == 0) {
        cout << "step 12 : m_(4,3) = m_(5,4).T().D() x m_(4,3)" << endl;
    }
    double t12_start = MPI_Wtime();
    diagonalizeV(m_54, dm, Column);
    multDimApplyPrune(m_43, dm, Row, false);
    double t12_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 12 (Total) : " << (t12_end - t12_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 13
    if (myrank == 0) {
        cout << "step 13 : m_(3,0) = m_(4,3).T().D() x m_(3,0)" << endl;
    }
    double t13_start = MPI_Wtime();
    diagonalizeV(m_43, dm, Column);
    multDimApplyPrune(m_30, dm, Row, false);
    double t13_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 13 (Total) : " << (t13_end - t13_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    printReducedInfo(m_30);

    double resgen_start = MPI_Wtime();
    resgen_l1(m_30, m_43, m_24, m_54, m_15, m_65);
    double resgen_end = MPI_Wtime();

    // end count time
    double total_computing_2 = MPI_Wtime();

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

    double t1_start = MPI_Wtime();

    clear_result_time();

    auto commGrid = m_10.getcommgrid();

    // m_10 becoms m_01
    m_10.Transpose();

    vector<IndexType> indl, indr, indj;

    vector<IndexType> order1 = {0, 0, 0, 1, 1, 0};

    double t1_end = MPI_Wtime();
    if (myrank == 0) {
        cout << "begin result generation ......" << endl;
        cout << "\ttranspose matrix and declarations take : " << (t1_end - t1_start) << " s\n" << endl;
    }

    get_local_indices(m_10, indl);
    send_local_indices(commGrid, indl);
    m_10.FreeMemory();

    get_local_indices(m_21, indr);
    send_local_indices(commGrid, indr);
    m_21.FreeMemory();

    local_join(commGrid, indl, indr, 2, 2, 1, 1, order1, indj);

    send_local_results(commGrid, indj.size() / 3);
}

void lubm10240_l2(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    clear_query_time();

    auto commWorld = G.getcommgrid();

    FullyDistVec<IndexType, ElementType> dm(commWorld);

    auto m_10(G), m_21(tG);

    FullyDistVec<IndexType, ElementType> r_10(commWorld, G.getnrow(), 0);
    r_10.SetElement(22773455, 2);

    // start count time
    double total_computing_1 = MPI_Wtime();

    // ==> step 1
    if (myrank == 0) {
        cout << "\n###############################################################" << endl;
        cout << "Query 2" << endl;
        cout << "###############################################################" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "step 1 : m_(1,0) = G x {1@(22773455,22773455)}*2" << endl;
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
        cout << "step 2 : m_(2,1) = G.T() * m_(1,0).D()*9" << endl;
    }
    double t2_start = MPI_Wtime();
    diagonalizeV(m_10, dm, Row, 9);
    multDimApplyPrune(m_21, dm, Column, true);
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
    diagonalizeV(m_21, dm, Column);
    multDimApplyPrune(m_10, dm, Row, false);
    double t3_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 3 (Total) : " << (t3_end - t3_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    printReducedInfo(m_10);

    double resgen_start = MPI_Wtime();
    resgen_l2(m_10, m_21);
    double resgen_end = MPI_Wtime();

    // end count time
    double total_computing_2 = MPI_Wtime();
    if (myrank == 0) {
        cout << "query2 mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "query2 prune time : " << total_prune_time << " s" << endl;
        cout << "query2 diag_reduce time : " << total_reduce_time << " s" << endl;
        cout << "query2 dim_apply time : " << total_dim_apply_time << " s" << endl;
        cout << "query2 result_enum time : " << resgen_end - resgen_start << " s" << endl;
        cout << "query2 time (Total) : " << total_computing_2 - total_computing_1 << " s" << endl;
    }
}

void lubm10240_l3(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    clear_query_time();

    auto commWorld = G.getcommgrid();

    FullyDistVec<IndexType, ElementType> dm(commWorld);

    auto m_30(G), m_43(tG), m_24(tG), m_54(G), m_15(tG), m_65(G);

    FullyDistVec<IndexType, ElementType> r_30(commWorld, G.getnrow(), 0), l_24(commWorld, G.getnrow(), 0), l_15(commWorld, G.getnrow(), 0);
    r_30.SetElement(20011736, 2);
    l_24.SetElement(24807258, 1);
    l_15.SetElement(23321407, 1);

    // start count time
    double total_computing_1 = MPI_Wtime();

    // ==> step 1
    if (myrank == 0) {
        cout << "\n###############################################################" << endl;
        cout << "Query 3" << endl;
        cout << "###############################################################" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "step 1 : m_(3,0) = G x {1@(20011736,20011736)}*2" << endl;
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
        cout << "step 2 : m_(4,3) = G.T() * m_(3, 0).D()*8" << endl;
    }
    double t2_start = MPI_Wtime();
    diagonalizeV(m_30, dm, Row, 8);
    multDimApplyPrune(m_43, dm, Column, true);
    double t2_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 2 (Total) : " << (t2_end - t2_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 3
    if (myrank == 0) {
        cout << "step 3 : m_(2,4) = G.T() x m_(4,3).D()*2" << endl;
    }
    double t3_start = MPI_Wtime();
    diagonalizeV(m_43, dm, Row, 2);
    multDimApplyPrune(m_24, dm, Column, true);
    double t3_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 3 (Total) : " << (t3_end - t3_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 4
    if (myrank == 0) {
        cout << "step 4 : m_(2,4) = {1@(24807258,24807258)} x m_(2,4)" << endl;
    }
    double t4_start = MPI_Wtime();
    multDimApplyPrune(m_24, l_24, Row, false);
    double t4_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 4 (Total) : " << (t4_end - t4_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 5
    if (myrank == 0) {
        cout << "step 5 : m_(5,4) = G x m_(2,4).T().D()*6" << endl;
    }
    double t5_start = MPI_Wtime();
    diagonalizeV(m_24, dm, Column, 6);
    multDimApplyPrune(m_54, dm, Column, true);
    double t5_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 5 (Total) : " << (t5_end - t5_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 6
    if (myrank == 0) {
        cout << "step 6 : m_(1,5) = G.T() x m_(5,4).D()*2" << endl;
    }
    double t6_start = MPI_Wtime();
    diagonalizeV(m_54, dm, Row, 2);
    multDimApplyPrune(m_15, dm, Column, true);
    double t6_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 6 (Total) : " << (t6_end - t6_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 7
    if (myrank == 0) {
        cout << "step 7 : m_(1,5) = {1@(23321407,23321407)} x m_(1,5)" << endl;
    }
    double t7_start = MPI_Wtime();
    multDimApplyPrune(m_15, l_15, Row, false);
    double t7_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 7 (Total) : " << (t7_end - t7_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 8
    if (myrank == 0) {
        cout << "step 8 : m_(6,5) = G x m_(1,5).T().D()*15" << endl;
    }
    double t8_start = MPI_Wtime();
    diagonalizeV(m_15, dm, Column, 15);
    multDimApplyPrune(m_65, dm, Column, true);
    double t8_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 8 (Total) : " << (t8_end - t8_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 9
    if (myrank == 0) {
        cout << "step 9 : m_(6,5) = m_(4,3).T().D() x m_(6,5)" << endl;
    }
    double t9_start = MPI_Wtime();
    diagonalizeV(m_43, dm, Column);
    multDimApplyPrune(m_65, dm, Row, false);
    double t9_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 9 (Total) : " << (t9_end - t9_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 10
    if (myrank == 0) {
        cout << "step 10 : m_(4,3) = m_(4,3) x m_(6,5).D()" << endl;
    }
    double t10_start = MPI_Wtime();
    diagonalizeV(m_65, dm);
    multDimApplyPrune(m_43, dm, Column, false);
    double t10_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 10 (Total) : " << (t10_end - t10_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 11
    if (myrank == 0) {
        cout << "step 11 : m_(5,4) = m_(6,5).T().D() x m_(5,4)" << endl;
    }
    double t11_start = MPI_Wtime();
    diagonalizeV(m_65, dm, Column);
    multDimApplyPrune(m_54, dm, Row, false);
    double t11_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 11 (Total) : " << (t11_end - t11_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 12
    if (myrank == 0) {
        cout << "step 12 : m_(4,3) = m_(5,4).T().D() x m_(4,3)" << endl;
    }
    double t12_start = MPI_Wtime();
    diagonalizeV(m_54, dm, Column);
    multDimApplyPrune(m_43, dm, Row, false);
    double t12_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 12 (Total) : " << (t12_end - t12_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 13
    if (myrank == 0) {
        cout << "step 13 : m_(3,0) = m_(4,3).T().D() x m_(3,0)" << endl;
    }
    double t13_start = MPI_Wtime();
    diagonalizeV(m_43, dm, Column);
    multDimApplyPrune(m_30, dm, Row, false);
    double t13_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 13 (Total) : " << (t13_end - t13_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    printReducedInfo(m_30);

    double resgen_start = MPI_Wtime();
    if (myrank == 0) {
            cout << "begin result generation ......" << endl;
            cout << "final size : 0" << endl;
            cout << "---------------------------------------------------------------" << endl;
    }
    double resgen_end = MPI_Wtime();

    // end count time
    double total_computing_2 = MPI_Wtime();
    if (myrank == 0) {
        cout << "query3 mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "query3 prune time : " << total_prune_time << " s" << endl;
        cout << "query3 diag_reduce time : " << total_reduce_time << " s" << endl;
        cout << "query3 dim_apply time : " << total_dim_apply_time << " s" << endl;
        cout << "query3 result_enum time : " << resgen_end - resgen_start << " s" << endl;
        cout << "query3 time (Total) : " << total_computing_2 - total_computing_1 << " s" << endl;
    }
}

void resgen_l4(PSpMat::MPI_DCCols &m_20, PSpMat::MPI_DCCols &m_52, PSpMat::MPI_DCCols &m_42, PSpMat::MPI_DCCols &m_32,
               PSpMat::MPI_DCCols &m_12) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t1_start = MPI_Wtime();

    clear_result_time();

    auto commGrid = m_20.getcommgrid();

    // m_20 becoms m_02
    m_20.Transpose();

    vector<IndexType> indl, indr, indj;
    
    vector<IndexType> order1 = {0, 0, 0, 1, 1, 0}, order2 = {0, 0, 0, 1, 1, 0, 0, 2}, order3 = {0, 0, 0, 1, 1, 0, 0, 2, 0, 3}, order4 = {0, 0, 1, 0, 0, 1, 0, 2, 0, 3, 0, 4};

    double t1_end = MPI_Wtime();
    if (myrank == 0) {
        cout << "begin result generation ......" << endl;
        cout << "\ttranspose matrix and declarations take : " << (t1_end - t1_start) << " s\n" << endl;
    }

    get_local_indices(m_20, indl);
    send_local_indices(commGrid, indl);
    m_20.FreeMemory();

    get_local_indices(m_52, indr);
    send_local_indices(commGrid, indr);
    m_52.FreeMemory();

    local_join(commGrid, indl, indr, 2, 2, 1, 1, order1, indj);
    indl = indj;

    get_local_indices(m_42, indr);
    send_local_indices(commGrid, indr);
    m_42.FreeMemory();

    local_join(commGrid, indl, indr, 3, 2, 1, 1, order2, indj);
    indl = indj;

    get_local_indices(m_32, indr);
    send_local_indices(commGrid, indr);
    m_32.FreeMemory();

    local_join(commGrid, indl, indr, 4, 2, 1, 1, order3, indj);
    indl = indj;

    get_local_indices(m_12, indr);
    send_local_indices(commGrid, indr);
    m_12.FreeMemory();

    local_join(commGrid, indl, indr, 5, 2, 1, 1, order4, indj);

    send_local_results(commGrid, indj.size() / 6);
}

void lubm10240_l4(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    clear_query_time();

    auto commWorld = G.getcommgrid();

    FullyDistVec<IndexType, ElementType> dm(commWorld);

    auto m_20(G), m_12(tG), m_32(tG), m_42(tG), m_52(tG);

    FullyDistVec<IndexType, ElementType> r_20(commWorld, G.getnrow(), 0), l_12(commWorld, G.getnrow(), 0);
    r_20.SetElement(10099852, 16);
    l_12.SetElement(10740785, 1);

    // start count time
    double total_computing_1 = MPI_Wtime();

    // ==> step 1
    if (myrank == 0) {
        cout << "\n###############################################################" << endl;
        cout << "Query 4" << endl;
        cout << "###############################################################" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "step 1 : m_(2,0) = G x {1@(10099852,10099852)}*16" << endl;
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
        cout << "step 2 : m_(1,2) = G.T() x m_(2,0).D()*2" << endl;
    }
    double t2_start = MPI_Wtime();
    diagonalizeV(m_20, dm, Row, 2);
    multDimApplyPrune(m_12, dm, Column, true);
    double t2_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 2 (Total) : " << (t2_end - t2_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 3
    if (myrank == 0) {
        cout << "step 3 : m_(1,2) = {1@(10740785,10740785)} x m_(1,2)" << endl;
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
        cout << "step 4 : m_(3,2) = G.T() x m_(1,2).T().D()*9" << endl;
    }
    double t4_start = MPI_Wtime();
    diagonalizeV(m_12, dm, Column, 9);
    multDimApplyPrune(m_32, dm, Column, true);
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
    diagonalizeV(m_32, dm, Column, 12);
    multDimApplyPrune(m_42, dm, Column, true);
    double t5_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 5 (Total) : " << (t5_end - t5_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 6
    if (myrank == 0) {
        cout << "step 6 : m_(5,2) = G.T() x m_(4,2).T().D()*17" << endl;
    }
    double t6_start = MPI_Wtime();
    diagonalizeV(m_42, dm, Column, 17);
    multDimApplyPrune(m_52, dm, Column, true);
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
    diagonalizeV(m_52, dm, Column);
    multDimApplyPrune(m_20, dm, Row, true);
    double t7_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 7 (Total) : " << (t7_end - t7_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    printReducedInfo(m_20);

    double resgen_start = MPI_Wtime();
    resgen_l4(m_20, m_52, m_42, m_32, m_12);
    double resgen_end = MPI_Wtime();

    // end count time
    double total_computing_2 = MPI_Wtime();
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

    double t1_start = MPI_Wtime();

    clear_result_time();

    auto commGrid = m_20.getcommgrid();

    // m_20 becoms m_02
    m_20.Transpose();

    vector<IndexType> indl, indr, indj;

    vector<IndexType> order1 = {0, 1, 1, 1, 0, 0};

    double t1_end = MPI_Wtime();
    if (myrank == 0) {
        cout << "begin result generation ......" << endl;
        cout << "\ttranspose matrix and declarations take : " << (t1_end - t1_start) << " s\n" << endl;
    }

    get_local_indices(m_20, indl);
    send_local_indices(commGrid, indl);
    m_20.FreeMemory();

    get_local_indices(m_12, indr);
    send_local_indices(commGrid, indr);
    m_12.FreeMemory();

    local_join(commGrid, indl, indr, 2, 2, 1, 1, order1, indj);

    send_local_results(commGrid, indj.size() / 3);
}

void lubm10240_l5(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    clear_query_time();

    auto commWorld = G.getcommgrid();

    FullyDistVec<IndexType, ElementType> dm(commWorld);
    
    auto m_20(G), m_12(tG);

    FullyDistVec<IndexType, ElementType> r_20(commWorld, G.getnrow(), 0), l_12(commWorld, G.getnrow(), 0);
    r_20.SetElement(10099852, 6);
    l_12.SetElement(2907611, 1);

    // start count time
    double total_computing_1 = MPI_Wtime();

    // ==> step 1
    if (myrank == 0) {
        cout << "\n###############################################################" << endl;
        cout << "Query 5" << endl;
        cout << "###############################################################" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "step 1 : m_(2,0) = G x {1@(10099852,10099852)}*6" << endl;
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
        cout << "step 2 : G.T() x m_(2,0).D()*2" << endl;
    }
    double t2_start = MPI_Wtime();
    diagonalizeV(m_20, dm, Row, 2);
    multDimApplyPrune(m_12, dm, Column, true);
    double t2_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 2 (Total) : " << (t2_end - t2_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 3
    if (myrank == 0) {
        cout << "step 3 : m_(1,2) = {1@(2907611,2907611)} x m_(1,2)" << endl;
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
    diagonalizeV(m_12, dm, Column);
    multDimApplyPrune(m_20, dm, Row, true);
    double t4_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 4 (Total) : " << (t4_end - t4_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    printReducedInfo(m_20);

    double resgen_start = MPI_Wtime();
    resgen_l5(m_20, m_12);
    double resgen_end = MPI_Wtime();

    // end count time
    double total_computing_2 = MPI_Wtime();
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

    double t1_start = MPI_Wtime();

    clear_result_time();

    auto commGrid = m_30.getcommgrid();

    // m_30 becoms m_03
    m_30.Transpose();

    vector<IndexType> indl, indr, indj;

    vector<IndexType> order1 = {0, 0, 0, 1, 2, 0}, order2 = {0, 0, 1, 0, 0, 1, 0, 2}, order3 = {0, 0, 1, 0, 0, 1, 0, 2, 0, 3};

    double t1_end = MPI_Wtime();
    if (myrank == 0) {
        cout << "begin result generation ......" << endl;
        cout << "\ttranspose matrix and declarations take : " << (t1_end - t1_start) << " s\n" << endl;
    }

    get_local_indices(m_30, indl);
    send_local_indices(commGrid, indl);
    m_30.FreeMemory();

    get_local_indices(m_43, indr);
    send_local_indices(commGrid, indr);
    m_43.FreeMemory();

    local_join(commGrid, indl, indr, 2, 2, 1, 1, order1, indj);
    local_redistribution(m_14, indj, 3, 2, indl);

    get_local_indices(m_24, indr);
    send_local_indices(commGrid, indr);
    m_24.FreeMemory();

    local_join(commGrid, indl, indr, 3, 2, 2, 1, order2, indj);
    indl = indj;

    get_local_indices(m_14, indr);
    send_local_indices(commGrid, indr);
    m_14.FreeMemory();

    local_join(m_14.getcommgrid(), indl, indr, 4, 2, 3, 1, order3, indj);

    send_local_results(commGrid, indj.size() / 5);
}

void lubm10240_l6(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    clear_query_time();

    auto commWorld = G.getcommgrid();

    FullyDistVec<IndexType, ElementType> dm(commWorld);
    
    auto m_30(G), m_43(tG), m_14(tG), m_24(tG);

    FullyDistVec<IndexType, ElementType> r_30(commWorld, G.getnrow(), 0), l_14(commWorld, G.getnrow(), 0), l_24(
            commWorld, G.getnrow(), 0);
    r_30.SetElement(10740785, 2);
    l_14.SetElement(13691764, 1);
    l_24.SetElement(23321407, 1);

    // start count time
    double total_computing_1 = MPI_Wtime();

    // ==> step 1
    if (myrank == 0) {
        cout << "\n###############################################################" << endl;
        cout << "Query 6" << endl;
        cout << "###############################################################" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "step 1 : m_(3,0) = G x {1@(10740785,10740785)}*2" << endl;
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
        cout << "step 2 : m_(4,3) = G.T() x m_(3,0).D()*16" << endl;
    }
    double t2_start = MPI_Wtime();
    diagonalizeV(m_30, dm, Row, 16);
    multDimApplyPrune(m_43, dm, Column, true);
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
    diagonalizeV(m_43, dm, Row, 6);
    multDimApplyPrune(m_14, dm, Column, true);
    double t3_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 3 (Total) : " << (t3_end - t3_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 4
    if (myrank == 0) {
        cout << "step 4 : m_(1,4) = {1@(13691764,13691764)} x m_(1,4)" << endl;
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
        cout << "step 5 : m_(2,4) = G.T() x m_(1,4).T().D()*2" << endl;
    }
    double t5_start = MPI_Wtime();
    diagonalizeV(m_14, dm, Column, 2);
    multDimApplyPrune(m_24, dm, Column, true);
    double t5_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 5 (Total) : " << (t5_end - t5_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 6
    if (myrank == 0) {
        cout << "step 6 : m_(2,4) = {1@(23321407,23321407)} x m_(2,4)" << endl;
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
    diagonalizeV(m_24, dm, Column);
    multDimApplyPrune(m_43, dm, Row, true);
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
    diagonalizeV(m_43, dm, Column);
    multDimApplyPrune(m_30, dm, Row, true);
    double t8_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 8 (Total) : " << (t8_end - t8_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    printReducedInfo(m_30);

    double resgen_start = MPI_Wtime();
    resgen_l6(m_30, m_43, m_14, m_24);
    double resgen_end = MPI_Wtime();

    // end count time
    double total_computing_2 = MPI_Wtime();
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

    double t1_start = MPI_Wtime();

    clear_result_time();

    auto commGrid = m_50.getcommgrid();

    // m_50 becoms m_05
    m_50.Transpose();

    vector<IndexType> indl, indr, indj;

    vector<IndexType> order1 = {0, 0, 1, 0, 1, 1}, order2 = {0, 0, 0, 1, 1, 0, 0, 2}, order3 = {0, 0, 0, 1, 0, 2, 0, 3},
                        order4 = {0, 0, 1, 0, 0, 1, 0, 2, 0, 3}, order5 = {0, 0, 1, 0, 0, 1, 0, 2, 0, 3, 0, 4};

    double t1_end = MPI_Wtime();
    if (myrank == 0) {
        cout << "begin result generation ......" << endl;
        cout << "\ttranspose matrix and declarations take : " << (t1_end - t1_start) << " s\n" << endl;
    }

    get_local_indices(m_50, indl);
    send_local_indices(commGrid, indl);
    m_50.FreeMemory();

    get_local_indices(m_35, indr);
    send_local_indices(commGrid, indr);
    m_35.FreeMemory();

    local_join(commGrid, indl, indr, 2, 2, 1, 1, order1, indj);
    local_redistribution(m_43, indj, 3, 1, indl);

    get_local_indices(m_43, indr);
    send_local_indices(commGrid, indr);
    m_43.FreeMemory();

    local_join(commGrid, indl, indr, 3, 2, 1, 1, order2, indj);
    local_redistribution(m_64, indj, 4, 2, indl);

    get_local_indices(m_64, indr);
    send_local_indices(commGrid, indr);
    m_64.FreeMemory();

    local_filter(commGrid, indl, indr, 4, 2, 2, 3, 1, 0, order3, indj);
    indl = indj;

    get_local_indices(m_24, indr);
    send_local_indices(commGrid, indr);
    m_64.FreeMemory();

    local_join(commGrid, indl, indr, 4, 2, 2, 1, order4, indj);
    local_redistribution(m_13, indj, 5, 2, indl);

    get_local_indices(m_13, indr);
    send_local_indices(commGrid, indr);
    m_13.FreeMemory();

    local_join(commGrid, indl, indr, 5, 2, 2, 1, order5, indj);

    send_local_results(commGrid, indj.size() / 6);
}

void lubm10240_l7(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    clear_query_time();

    auto commWorld = G.getcommgrid();

    FullyDistVec<IndexType, ElementType> dm(commWorld);
    
    auto m_50(G), m_35(G), m_13(tG), m_43(tG), m_24(tG), m_64(G);

    FullyDistVec<IndexType, ElementType> r_50(commWorld, G.getnrow(), 0), l_13(commWorld, G.getnrow(), 0), l_24(
            commWorld, G.getnrow(), 0);
    r_50.SetElement(10740785, 2);
    l_13.SetElement(20011736, 1);
    l_24.SetElement(22773455, 1);

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
    diagonalizeV(m_50, dm, Row, 18);
    multDimApplyPrune(m_35, dm, Column, true);
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
    diagonalizeV(m_35, dm, Row, 2);
    multDimApplyPrune(m_13, dm, Column, true);
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
    diagonalizeV(m_13, dm, Column, 14);
    multDimApplyPrune(m_43, dm, Column, true);
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
    diagonalizeV(m_43, dm, Row, 2);
    multDimApplyPrune(m_24, dm, Column, true);
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
    diagonalizeV(m_24, dm, Column, 13);
    multDimApplyPrune(m_64, dm, Column, true);
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
    diagonalizeV(m_35, dm, Column);
    multDimApplyPrune(m_64, dm, Row, false);
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
    diagonalizeV(m_64, dm);
    multDimApplyPrune(m_35, dm, Column, false);
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
    diagonalizeV(m_64, dm, Column);
    multDimApplyPrune(m_43, dm, Row, false);
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
    diagonalizeV(m_43, dm, Column);
    multDimApplyPrune(m_35, dm, Row, false);
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
    diagonalizeV(m_35, dm, Column);
    multDimApplyPrune(m_50, dm, Row, false);
    double t13_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 13 (Total) : " << (t13_end - t13_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    printReducedInfo(m_50);

    double resgen_start = MPI_Wtime();
    resgen_l7(m_50, m_35, m_43, m_64, m_24, m_13);
    double resgen_end = MPI_Wtime();

    // end count time
    double total_computing_2 = MPI_Wtime();
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

    // set comparasion function pointer array, for qsort in result generation
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
            cout << "Usage: ./encoded_lubm10240" << endl;
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

        PSpMat::MPI_DCCols G(MPI_COMM_WORLD);
        auto commWorld = G.getcommgrid();

        string Mname("/scratch/cheny0l/trill_exp/paracoder/paracoder_output/lubm1B_mm_ready/paracoder_lubm1B.mm");

        double t1 = MPI_Wtime();
        G.ParallelReadMM(Mname, true, selectSecond);
        double t2 = MPI_Wtime();

        G.PrintInfo();
        float imG = G.LoadImbalance();

        if (myrank == 0) {
            cout << "\tread file takes : " << (t2 - t1) << " s" << endl;
            cout << "\toriginal imbalance of G : " << imG << endl;
        }

        double t1_trans = MPI_Wtime();
        auto tG = transpose(G);
        double t2_trans = MPI_Wtime();

        if (myrank == 0) {
            cout << "\ttranspose G takes : " << (t2_trans - t1_trans) << " s" << endl;
            cout << "graph load (Total) : " << (t2_trans - t1) << " s" << endl;
            cout << "---------------------------------------------------------------" << endl;
        }

        // run 7 queries 5 times each
       for (int time = 1; time <= 5; time++) {
            lubm10240_l1(G, tG);
            lubm10240_l2(G, tG);
            lubm10240_l3(G, tG);
            lubm10240_l4(G, tG);
            lubm10240_l5(G, tG);
            lubm10240_l6(G, tG);
            lubm10240_l7(G, tG);
       }
    }

    MPI_Finalize();
    return 0;
}