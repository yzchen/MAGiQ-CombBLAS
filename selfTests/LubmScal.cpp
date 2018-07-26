#include <iostream>
#include <functional>
#include <algorithm>
#include <sstream>
#include <vector>
#include <iterator>
#include <fstream>
#include "../selfInclude/HeaderScal.h"

using namespace std;
using namespace combblas;

void resgen_l1(PSpMat::MPI_DCCols &m_40, PSpMat::MPI_DCCols &m_34, PSpMat::MPI_DCCols &m_23, PSpMat::MPI_DCCols &m_53,
               PSpMat::MPI_DCCols &m_15, PSpMat::MPI_DCCols &m_65) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t1_start = MPI_Wtime();
    clear_result_time();

    auto commGrid = m_40.getcommgrid();

    // m_40 becoms m_04
    m_40.Transpose();

    vector<IndexType> indl, indr, indj;

    vector<IndexType> order1 = {0, 1, 1, 0, 1, 1}, order2 = {0, 0, 0, 1, 0, 2, 1, 0}, order3 = {0, 0, 0, 1, 0, 2, 0, 3},
            order4 = {0, 0, 1, 0, 0, 1, 0, 2, 0, 3}, order5 = {0, 0, 0, 1, 1, 0, 0, 2, 0, 3, 0, 4};

    double t1_end = MPI_Wtime();
    if (myrank == 0) {
        cout << "begin result generation ......" << endl;
        cout << "\ttranspose matrix and declarations take : " << (t1_end - t1_start) << " s\n" << endl;
    }

    get_local_indices(m_40, indl);
    send_local_indices(commGrid, indl);
    // m_40.FreeMemory();

    get_local_indices(m_34, indr);
    send_local_indices(commGrid, indr);
    // m_34.FreeMemory();

    local_join(commGrid, indl, indr, 2, 2, 1, 1, order1, indj);
    local_redistribution(m_53, indj, 3, 1, indl);

    get_local_indices(m_53, indr);
    send_local_indices(commGrid, indr);
    // m_53.FreeMemory();

    local_join(commGrid, indl, indr, 3, 2, 1, 1, order2, indj);
    local_redistribution(m_65, indj, 4, 3, indl);

    get_local_indices(m_65, indr);
    send_local_indices(commGrid, indr);
    // m_65.FreeMemory();

    local_filter(commGrid, indl, indr, 4, 2, 3, 2, 1, 0, order3, indj);
    indl = indj;

    get_local_indices(m_15, indr);
    send_local_indices(commGrid, indr);
    // m_15.FreeMemory();

    local_join(commGrid, indl, indr, 4, 2, 3, 1, order4, indj);
    local_redistribution(m_23, indj, 5, 2, indl);

    get_local_indices(m_23, indr);
    send_local_indices(commGrid, indr);
    // m_23.FreeMemory();

    local_join(commGrid, indl, indr, 5, 2, 2, 1, order5, indj);
    send_local_results(commGrid, indj.size() / 6);
}

void lubm_l1(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG, FullyDistVec<IndexType, IndexType> &nonisov) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    clear_query_time();

    auto commWorld = G.getcommgrid();

    FullyDistVec<IndexType, ElementType> dm(commWorld);

    auto m_40(G), m_34(G), m_23(tG), m_53(tG), m_15(tG), m_65(tG);

    IndexType ind1 = nonisov.FindInds(std::bind2nd(std::equal_to<IndexType>(), static_cast<IndexType>(103594630)))[0];
    IndexType ind2 = nonisov.FindInds(std::bind2nd(std::equal_to<IndexType>(), static_cast<IndexType>(139306106)))[0];
    IndexType ind3 = nonisov.FindInds(std::bind2nd(std::equal_to<IndexType>(), static_cast<IndexType>(130768016)))[0];

    FullyDistVec<IndexType, ElementType> r_40(commWorld, G.getnrow(), 0), l_23(commWorld, G.getnrow(), 0), l_15(commWorld, G.getnrow(), 0);
    r_40.SetElement(ind1, 17);
    l_23.SetElement(ind2, 1);
    l_15.SetElement(ind3, 1);

    // start count time
    double total_computing_1 = MPI_Wtime();

    // ==> step 1
    if (myrank == 0) {
        cout << "\n###############################################################" << endl;
        cout << "Query 1" << endl;
        cout << "###############################################################" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "step 1 : m_(4,0) = G x {1@(103594630,103594630)}*17" << endl;
    }
    double t1_start = MPI_Wtime();
    multDimApplyPrune(m_40, r_40, Column, true);
    m_40.PrintInfo();
    double t1_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 1 (Total) : " << (t1_end - t1_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 2
    if (myrank == 0) {
        cout << "step 2 : m_(3,4) = G x m_(4, 0).D()*12" << endl;
    }
    double t2_start = MPI_Wtime();
    diagonalizeV(m_40, dm, Row, 12);
    multDimApplyPrune(m_34, dm, Column, true);
    m_34.PrintInfo();
    double t2_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 2 (Total) : " << (t2_end - t2_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 3
    if (myrank == 0) {
        cout << "step 3 : m_(2,3) = G.T() x m_(3,4).D()*17" << endl;
    }
    double t3_start = MPI_Wtime();
    diagonalizeV(m_34, dm, Row, 17);
    multDimApplyPrune(m_23, dm, Column, true);
    m_23.PrintInfo();
    double t3_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 3 (Total) : " << (t3_end - t3_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 4
    if (myrank == 0) {
        cout << "step 4 : m_(2,3) = {1@(139306106,139306106)} x m_(2,3)" << endl;
    }
    double t4_start = MPI_Wtime();
    multDimApplyPrune(m_23, l_23, Row, false);
    m_23.PrintInfo();
    double t4_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 4 (Total) : " << (t4_end - t4_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 5
    if (myrank == 0) {
        cout << "step 5 : m_(5,3) = G x m_(2,3).T().D()*14" << endl;
    }
    double t5_start = MPI_Wtime();
    diagonalizeV(m_23, dm, Column, 14);
    multDimApplyPrune(m_53, dm, Column, true);
    m_53.PrintInfo();
    double t5_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 5 (Total) : " << (t5_end - t5_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 6
    if (myrank == 0) {
        cout << "step 6 : m_(1,5) = G.T() x m_(5,3).D()*17" << endl;
    }
    double t6_start = MPI_Wtime();
    diagonalizeV(m_53, dm, Row, 17);
    multDimApplyPrune(m_15, dm, Column, true);
    m_15.PrintInfo();
    double t6_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 6 (Total) : " << (t6_end - t6_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 7
    if (myrank == 0) {
        cout << "step 7 : m_(1,5) = {1@(130768016,130768016)} x m_(1,5)" << endl;
    }
    double t7_start = MPI_Wtime();
    multDimApplyPrune(m_15, l_15, Row, false);
    m_15.PrintInfo();
    double t7_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 7 (Total) : " << (t7_end - t7_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 8
    if (myrank == 0) {
        cout << "step 8 : m_(6,5) = G.T() x m_(1,5).T().D()*3" << endl;
    }
    double t8_start = MPI_Wtime();
    diagonalizeV(m_15, dm, Column, 3);
    multDimApplyPrune(m_65, dm, Column, true);
    m_65.PrintInfo();
    double t8_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 8 (Total) : " << (t8_end - t8_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 9
    if (myrank == 0) {
        cout << "step 9 : m_(6,5) = m_(3,4).T().D() x m_(6,5)" << endl;
    }
    double t9_start = MPI_Wtime();
    diagonalizeV(m_34, dm, Column);
    multDimApplyPrune(m_65, dm, Row, false);
    m_65.PrintInfo();
    double t9_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 9 (Total) : " << (t9_end - t9_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 10
    if (myrank == 0) {
        cout << "step 10 : m_(3,4) = m_(3,4) x m_(6,5).D()" << endl;
    }
    double t10_start = MPI_Wtime();
    diagonalizeV(m_65, dm, Row);
    multDimApplyPrune(m_34, dm, Column, false);
    m_34.PrintInfo();
    double t10_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 10 (Total) : " << (t10_end - t10_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 11
    if (myrank == 0) {
        cout << "step 11 : m_(5,3) = m_(6,5).T().D() x m_(5,3)" << endl;
    }
    double t11_start = MPI_Wtime();
    diagonalizeV(m_65, dm, Column);
    multDimApplyPrune(m_53, dm, Row, false);
    m_53.PrintInfo();
    double t11_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 11 (Total) : " << (t11_end - t11_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 12
    if (myrank == 0) {
        cout << "step 12 : m_(3,4) = m_(5,3).T().D() x m_(3,4)" << endl;
    }
    double t12_start = MPI_Wtime();
    diagonalizeV(m_53, dm, Column);
    multDimApplyPrune(m_34, dm, Row, false);
    m_34.PrintInfo();
    double t12_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 12 (Total) : " << (t12_end - t12_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 13
    if (myrank == 0) {
        cout << "step 13 : m_(4,0) = m_(3,4).T().D() x m_(4,0)" << endl;
    }
    double t13_start = MPI_Wtime();
    diagonalizeV(m_34, dm, Column);
    multDimApplyPrune(m_40, dm, Row, false);
    m_40.PrintInfo();
    double t13_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 13 (Total) : " << (t13_end - t13_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    double query_counting = MPI_Wtime();

    if (myrank == 0) {
        cout << "query1 mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "query1 prune time : " << total_prune_time << " s" << endl;
        cout << "query1 diag_reduce time : " << total_reduce_time << " s" << endl;
        cout << "query1 dim_apply time : " << total_dim_apply_time << " s" << endl;
        cout << "query1 total query execution time : " << query_counting - total_computing_1 << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    double resgen_start = MPI_Wtime();
    resgen_l1(m_40, m_34, m_23, m_53, m_15, m_65);
    double resgen_end = MPI_Wtime();

    // end count total time
    double total_computing_2 = MPI_Wtime();

    if (myrank == 0) {
        cout << "query1 result_enum time : " << resgen_end - resgen_start << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "query1 time (Total) : " << total_computing_2 - total_computing_1 << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
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

void lubm_l2(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG, FullyDistVec<IndexType, IndexType> &nonisov) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    clear_query_time();

    auto commWorld = G.getcommgrid();

    FullyDistVec<IndexType, ElementType> dm(commWorld);

    auto m_10(G), m_21(tG);

    IndexType ind1 = nonisov.FindInds(std::bind2nd(std::equal_to<IndexType>(), static_cast<IndexType>(235928023)))[0];

    FullyDistVec<IndexType, ElementType> r_10(commWorld, G.getnrow(), 0);
    r_10.SetElement(ind1, 17);

    // start count time
    double total_computing_1 = MPI_Wtime();

    // ==> step 1
    if (myrank == 0) {
        cout << "\n###############################################################" << endl;
        cout << "Query 2" << endl;
        cout << "###############################################################" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "step 1 : m_(1,0) = G x {1@(235928023,235928023)}*17" << endl;
    }
    double t1_start = MPI_Wtime();
    multDimApplyPrune(m_10, r_10, Column, true);
    m_10.PrintInfo();
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
    m_21.PrintInfo();
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
    m_10.PrintInfo();
    double t3_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 3 (Total) : " << (t3_end - t3_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    double query_counting = MPI_Wtime();

    if (myrank == 0) {
        cout << "query2 mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "query2 prune time : " << total_prune_time << " s" << endl;
        cout << "query2 diag_reduce time : " << total_reduce_time << " s" << endl;
        cout << "query2 dim_apply time : " << total_dim_apply_time << " s" << endl;
        cout << "query2 total query execution time : " << query_counting - total_computing_1 << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    double resgen_start = MPI_Wtime();
    resgen_l2(m_10, m_21);
    double resgen_end = MPI_Wtime();

    // end count time
    double total_computing_2 = MPI_Wtime();
    if (myrank == 0) {
        cout << "query2 result_enum time : " << resgen_end - resgen_start << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "query2 time (Total) : " << total_computing_2 - total_computing_1 << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }
}

void lubm_l3(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG, FullyDistVec<IndexType, IndexType> &nonisov) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    clear_query_time();

    auto commWorld = G.getcommgrid();

    FullyDistVec<IndexType, ElementType> dm(commWorld);

    auto m_40(G), m_34(G), m_23(tG), m_53(tG), m_15(tG), m_65(tG);

    IndexType ind1 = nonisov.FindInds(std::bind2nd(std::equal_to<IndexType>(), static_cast<IndexType>(103594630)))[0];
    IndexType ind2 = nonisov.FindInds(std::bind2nd(std::equal_to<IndexType>(), static_cast<IndexType>(223452631)))[0];
    IndexType ind3 = nonisov.FindInds(std::bind2nd(std::equal_to<IndexType>(), static_cast<IndexType>(130768016)))[0];

    FullyDistVec<IndexType, ElementType> r_40(commWorld, G.getnrow(), 0), l_23(commWorld, G.getnrow(), 0), l_15(commWorld, G.getnrow(), 0);
    r_40.SetElement(ind1, 17);
    l_23.SetElement(ind2, 1);
    l_15.SetElement(ind3, 1);

    // start count time
    double total_computing_1 = MPI_Wtime();

    // ==> step 1
    if (myrank == 0) {
        cout << "\n###############################################################" << endl;
        cout << "Query 3" << endl;
        cout << "###############################################################" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "step 1 : m_(4,0) = G x {1@(103594630,103594630)}*17" << endl;
    }
    double t1_start = MPI_Wtime();
    multDimApplyPrune(m_40, r_40, Column, true);
    m_40.PrintInfo();
    double t1_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 1 (Total) : " << (t1_end - t1_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 2
    if (myrank == 0) {
        cout << "step 2 : m_(3,4) = G x m_(4, 0).D()*12" << endl;
    }
    double t2_start = MPI_Wtime();
    diagonalizeV(m_40, dm, Row, 12);
    multDimApplyPrune(m_34, dm, Column, true);
    m_34.PrintInfo();
    double t2_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 2 (Total) : " << (t2_end - t2_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 3
    if (myrank == 0) {
        cout << "step 3 : m_(2,3) = G.T() x m_(3,4).D()*17" << endl;
    }
    double t3_start = MPI_Wtime();
    diagonalizeV(m_34, dm, Row, 17);
    multDimApplyPrune(m_23, dm, Column, true);
    m_23.PrintInfo();
    double t3_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 3 (Total) : " << (t3_end - t3_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 4
    if (myrank == 0) {
        cout << "step 4 : m_(2,3) = {1@(223452631,223452631)} x m_(2,3)" << endl;
    }
    double t4_start = MPI_Wtime();
    multDimApplyPrune(m_23, l_23, Row, false);
    m_23.PrintInfo();
    double t4_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 4 (Total) : " << (t4_end - t4_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

//    // ==> step 5
//    if (myrank == 0) {
//        cout << "step 5 : m_(5,3) = G x m_(2,3).T().D()*14" << endl;
//    }
//    double t5_start = MPI_Wtime();
//    diagonalizeV(m_23, dm, Column, 14);
//    multDimApplyPrune(m_53, dm, Column, true);
//    m_53.PrintInfo();
//    double t5_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 5 (Total) : " << (t5_end - t5_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }
//
//    // ==> step 6
//    if (myrank == 0) {
//        cout << "step 6 : m_(1,5) = G.T() x m_(5,3).D()*17" << endl;
//    }
//    double t6_start = MPI_Wtime();
//    diagonalizeV(m_53, dm, Row, 17);
//    multDimApplyPrune(m_15, dm, Column, true);
//    m_15.PrintInfo();
//    double t6_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 6 (Total) : " << (t6_end - t6_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }
//
//    // ==> step 7
//    if (myrank == 0) {
//        cout << "step 7 : m_(1,5) = {1@(130768016,130768016)} x m_(1,5)" << endl;
//    }
//    double t7_start = MPI_Wtime();
//    multDimApplyPrune(m_15, l_15, Row, false);
//    m_15.PrintInfo();
//    double t7_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 7 (Total) : " << (t7_end - t7_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }
//
//    // ==> step 8
//    if (myrank == 0) {
//        cout << "step 8 : m_(6,5) = G.T() x m_(1,5).T().D()*3" << endl;
//    }
//    double t8_start = MPI_Wtime();
//    diagonalizeV(m_15, dm, Column, 3);
//    multDimApplyPrune(m_65, dm, Column, true);
//    m_65.PrintInfo();
//    double t8_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 8 (Total) : " << (t8_end - t8_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }
//
//    // ==> step 9
//    if (myrank == 0) {
//        cout << "step 9 : m_(6,5) = m_(3,4).T().D() x m_(6,5)" << endl;
//    }
//    double t9_start = MPI_Wtime();
//    diagonalizeV(m_34, dm, Column);
//    multDimApplyPrune(m_65, dm, Row, false);
//    m_65.PrintInfo();
//    double t9_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 9 (Total) : " << (t9_end - t9_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }
//
//    // ==> step 10
//    if (myrank == 0) {
//        cout << "step 10 : m_(3,4) = m_(3,4) x m_(6,5).D()" << endl;
//    }
//    double t10_start = MPI_Wtime();
//    diagonalizeV(m_65, dm, Row);
//    multDimApplyPrune(m_34, dm, Column, false);
//    m_34.PrintInfo();
//    double t10_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 10 (Total) : " << (t10_end - t10_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }
//
//    // ==> step 11
//    if (myrank == 0) {
//        cout << "step 11 : m_(5,3) = m_(6,5).T().D() x m_(5,3)" << endl;
//    }
//    double t11_start = MPI_Wtime();
//    diagonalizeV(m_65, dm, Column);
//    multDimApplyPrune(m_53, dm, Row, false);
//    m_53.PrintInfo();
//    double t11_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 11 (Total) : " << (t11_end - t11_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }
//
//    // ==> step 12
//    if (myrank == 0) {
//        cout << "step 12 : m_(3,4) = m_(5,3).T().D() x m_(3,4)" << endl;
//    }
//    double t12_start = MPI_Wtime();
//    diagonalizeV(m_53, dm, Column);
//    multDimApplyPrune(m_34, dm, Row, false);
//    m_34.PrintInfo();
//    double t12_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 12 (Total) : " << (t12_end - t12_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }
//
//    // ==> step 13
//    if (myrank == 0) {
//        cout << "step 13 : m_(4,0) = m_(3,4).T().D() x m_(4,0)" << endl;
//    }
//    double t13_start = MPI_Wtime();
//    diagonalizeV(m_34, dm, Column);
//    multDimApplyPrune(m_40, dm, Row, false);
//    m_40.PrintInfo();
//    double t13_end = MPI_Wtime();
//
//    if (myrank == 0) {
//        cout << "step 13 (Total) : " << (t13_end - t13_start) << " s" << endl;
//        cout << "---------------------------------------------------------------" << endl;
//    }

    double query_counting = MPI_Wtime();
    if (myrank == 0) {
        cout << "query3 mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "query3 prune time : " << total_prune_time << " s" << endl;
        cout << "query3 diag_reduce time : " << total_reduce_time << " s" << endl;
        cout << "query3 dim_apply time : " << total_dim_apply_time << " s" << endl;
        cout << "query3 total query execution time : " << query_counting - total_computing_1 << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    double resgen_start = MPI_Wtime();
    if (myrank == 0) {
        cout << "final size : 0" << endl;
        cout << "total get local indices time : " << total_get_local_indices_time << " s" << endl;
        cout << "total send local indices time : " << total_send_local_indices_time << " s" << endl;
        cout << "total local join time : " << total_local_join_time << " s" << endl;
        cout << "total local filter time : " << total_local_filter_time << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }
    double resgen_end = MPI_Wtime();

    // end count time
    double total_computing_2 = MPI_Wtime();

    if (myrank == 0) {
        cout << "query3 result_enum time : " << resgen_end - resgen_start << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "query3 time (Total) : " << total_computing_2 - total_computing_1 << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
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

void lubm_l4(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG, FullyDistVec<IndexType, IndexType> &nonisov) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    clear_query_time();

    auto commWorld = G.getcommgrid();

    FullyDistVec<IndexType, ElementType> dm(commWorld);

    auto m_20(G), m_12(tG), m_32(tG), m_42(tG), m_52(tG);

    IndexType ind1 = nonisov.FindInds(std::bind2nd(std::equal_to<IndexType>(), static_cast<IndexType>(2808777)))[0];
    IndexType ind2 = nonisov.FindInds(std::bind2nd(std::equal_to<IndexType>(), static_cast<IndexType>(291959481)))[0];

    FullyDistVec<IndexType, ElementType> r_20(commWorld, G.getnrow(), 0), l_12(commWorld, G.getnrow(), 0);
    r_20.SetElement(ind1, 7);
    l_12.SetElement(ind2, 1);

    // start count time
    double total_computing_1 = MPI_Wtime();

    // ==> step 1
    if (myrank == 0) {
        cout << "\n###############################################################" << endl;
        cout << "Query 4" << endl;
        cout << "###############################################################" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "step 1 : m_(2,0) = G x {1@(2808777,2808777)}*7" << endl;
    }
    double t1_start = MPI_Wtime();
    multDimApplyPrune(m_20, r_20, Column, true);
    m_20.PrintInfo();
    double t1_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 1 (Total) : " << (t1_end - t1_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 2
    if (myrank == 0) {
        cout << "step 2 : m_(1,2) = G.T() x m_(2,0).D()*17" << endl;
    }
    double t2_start = MPI_Wtime();
    diagonalizeV(m_20, dm, Row, 17);
    multDimApplyPrune(m_12, dm, Column, true);
    m_12.PrintInfo();
    double t2_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 2 (Total) : " << (t2_end - t2_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 3
    if (myrank == 0) {
        cout << "step 3 : m_(1,2) = {1@(291959481,291959481)} x m_(1,2)" << endl;
    }
    double t3_start = MPI_Wtime();
    multDimApplyPrune(m_12, l_12, Row, false);
    m_12.PrintInfo();
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
    m_32.PrintInfo();
    double t4_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 4 (Total) : " << (t4_end - t4_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 5
    if (myrank == 0) {
        cout << "step 5 : m_(4,2) = G.T() x m_(3,2).T().D()*8" << endl;
    }
    double t5_start = MPI_Wtime();
    diagonalizeV(m_32, dm, Column, 8);
    multDimApplyPrune(m_42, dm, Column, true);
    m_42.PrintInfo();
    double t5_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 5 (Total) : " << (t5_end - t5_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 6
    if (myrank == 0) {
        cout << "step 6 : m_(5,2) = G.T() x m_(4,2).T().D()*2" << endl;
    }
    double t6_start = MPI_Wtime();
    diagonalizeV(m_42, dm, Column, 2);
    multDimApplyPrune(m_52, dm, Column, true);
    m_52.PrintInfo();
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
    m_20.PrintInfo();
    double t7_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 7 (Total) : " << (t7_end - t7_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    double query_counting = MPI_Wtime();

    if (myrank == 0) {
        cout << "query4 mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "query4 prune time : " << total_prune_time << " s" << endl;
        cout << "query4 diag_reduce time : " << total_reduce_time << " s" << endl;
        cout << "query4 dim_apply time : " << total_dim_apply_time << " s" << endl;
        cout << "query4 total query execution time : " << query_counting - total_computing_1 << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    double resgen_start = MPI_Wtime();
    resgen_l4(m_20, m_52, m_42, m_32, m_12);
    double resgen_end = MPI_Wtime();

    // end count time
    double total_computing_2 = MPI_Wtime();
    if (myrank == 0) {
        cout << "query4 result_enum time : " << resgen_end - resgen_start << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "query4 time (Total) : " << total_computing_2 - total_computing_1 << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
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

void lubm_l5(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG, FullyDistVec<IndexType, IndexType> &nonisov) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    clear_query_time();

    auto commWorld = G.getcommgrid();

    FullyDistVec<IndexType, ElementType> dm(commWorld);

    auto m_20(G), m_12(tG);

    IndexType ind1 = nonisov.FindInds(std::bind2nd(std::equal_to<IndexType>(), static_cast<IndexType>(191176245)))[0];
    IndexType ind2 = nonisov.FindInds(std::bind2nd(std::equal_to<IndexType>(), static_cast<IndexType>(2808777)))[0];

    FullyDistVec<IndexType, ElementType> r_20(commWorld, G.getnrow(), 0), l_12(commWorld, G.getnrow(), 0);
    r_20.SetElement(ind1, 17);
    l_12.SetElement(ind2, 1);

    // start count time
    double total_computing_1 = MPI_Wtime();

    // ==> step 1
    if (myrank == 0) {
        cout << "\n###############################################################" << endl;
        cout << "Query 5" << endl;
        cout << "###############################################################" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "step 1 : m_(2,0) = G x {1@(191176245,191176245)}*17" << endl;
    }
    double t1_start = MPI_Wtime();
    multDimApplyPrune(m_20, r_20, Column, true);
    m_20.PrintInfo();
    double t1_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 1 (Total) : " << (t1_end - t1_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 2
    if (myrank == 0) {
        cout << "step 2 : G.T() x m_(2,0).D()*3" << endl;
    }
    double t2_start = MPI_Wtime();
    diagonalizeV(m_20, dm, Row, 3);
    multDimApplyPrune(m_12, dm, Column, true);
    m_12.PrintInfo();
    double t2_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 2 (Total) : " << (t2_end - t2_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 3
    if (myrank == 0) {
        cout << "step 3 : m_(1,2) = {1@(2808777,2808777)} x m_(1,2)" << endl;
    }
    double t3_start = MPI_Wtime();
    multDimApplyPrune(m_12, l_12, Row, false);
    m_12.PrintInfo();
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
    m_20.PrintInfo();
    double t4_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 4 (Total) : " << (t4_end - t4_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    double query_counting = MPI_Wtime();

    if (myrank == 0) {
        cout << "query5 mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "query5 prune time : " << total_prune_time << " s" << endl;
        cout << "query5 diag_reduce time : " << total_reduce_time << " s" << endl;
        cout << "query5 dim_apply time : " << total_dim_apply_time << " s" << endl;
        cout << "query5 total query execution time : " << query_counting - total_computing_1 << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    double resgen_start = MPI_Wtime();
    resgen_l5(m_20, m_12);
    double resgen_end = MPI_Wtime();

    // end count time
    double total_computing_2 = MPI_Wtime();
    if (myrank == 0) {
        cout << "query5 result_enum time : " << resgen_end - resgen_start << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "query5 time (Total) : " << total_computing_2 - total_computing_1 << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }
}

void resgen_l6(PSpMat::MPI_DCCols &m_40, PSpMat::MPI_DCCols &m_14, PSpMat::MPI_DCCols &m_34, PSpMat::MPI_DCCols &m_23) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t1_start = MPI_Wtime();

    clear_result_time();

    auto commGrid = m_40.getcommgrid();

    // m_40 becoms m_04
    m_40.Transpose();

    vector<IndexType> indl, indr, indj;

    vector<IndexType> order1 = {0, 0, 1, 0, 1, 1}, order2 = {0, 0, 1, 0, 0, 1, 0, 2}, order3 = {0, 0, 1, 0, 0, 1, 0, 2, 0, 3};

    double t1_end = MPI_Wtime();
    if (myrank == 0) {
        cout << "begin result generation ......" << endl;
        cout << "\ttranspose matrix and declarations take : " << (t1_end - t1_start) << " s\n" << endl;
    }

    get_local_indices(m_40, indl);
    send_local_indices(commGrid, indl);
    m_40.FreeMemory();

    get_local_indices(m_34, indr);
    send_local_indices(commGrid, indr);
    m_34.FreeMemory();

    local_join(commGrid, indl, indr, 2, 2, 1, 1, order1, indj);
    local_redistribution(m_23, indj, 3, 1, indl);

    get_local_indices(m_23, indr);
    send_local_indices(commGrid, indr);
    m_23.FreeMemory();

    local_join(commGrid, indl, indr, 3, 2, 1, 1, order2, indj);
    local_redistribution(m_14, indj, 4, 3, indl);

    get_local_indices(m_14, indr);
    send_local_indices(commGrid, indr);
    m_14.FreeMemory();

    local_join(commGrid, indl, indr, 4, 2, 3, 1, order3, indj);
    send_local_results(commGrid, indj.size() / 5);
}

void lubm_l6(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG, FullyDistVec<IndexType, IndexType> &nonisov) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    clear_query_time();

    auto commWorld = G.getcommgrid();

    FullyDistVec<IndexType, ElementType> dm(commWorld);

    auto m_40(G), m_14(tG), m_34(G), m_23(tG);

    IndexType ind1 = nonisov.FindInds(std::bind2nd(std::equal_to<IndexType>(), static_cast<IndexType>(130768016)))[0];
    IndexType ind2 = nonisov.FindInds(std::bind2nd(std::equal_to<IndexType>(), static_cast<IndexType>(267261320)))[0];
    IndexType ind3 = nonisov.FindInds(std::bind2nd(std::equal_to<IndexType>(), static_cast<IndexType>(291959481)))[0];

    FullyDistVec<IndexType, ElementType> r_40(commWorld, G.getnrow(), 0), l_14(commWorld, G.getnrow(), 0), l_23(
            commWorld, G.getnrow(), 0);
    r_40.SetElement(ind1, 17);
    l_14.SetElement(ind2, 1);
    l_23.SetElement(ind3, 1);

    // start count time
    double total_computing_1 = MPI_Wtime();

    // ==> step 1
    if (myrank == 0) {
        cout << "\n###############################################################" << endl;
        cout << "Query 6" << endl;
        cout << "###############################################################" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "step 1 : m_(4,0) = G x {1@(130768016,130768016)}*17" << endl;
    }
    double t1_start = MPI_Wtime();
    multDimApplyPrune(m_40, r_40, Column, true);
    m_40.PrintInfo();
    double t1_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 1 (Total) : " << (t1_end - t1_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 2
    if (myrank == 0) {
        cout << "step 2 : m_(1,4) = G.T() x m_(4,0).D()*3" << endl;
    }
    double t2_start = MPI_Wtime();
    diagonalizeV(m_40, dm, Row, 3);
    multDimApplyPrune(m_14, dm, Column, true);
    m_14.PrintInfo();
    double t2_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 2 (Total) : " << (t2_end - t2_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 3
    if (myrank == 0) {
        cout << "step 3 : m_(1,4) = {1@(267261320,267261320)} x m_(1,4)" << endl;
    }
    double t3_start = MPI_Wtime();
    multDimApplyPrune(m_14, l_14, Row, false);
    m_14.PrintInfo();
    double t3_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 3 (Total) : " << (t3_end - t3_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 4
    if (myrank == 0) {
        cout << "step 4 : m_(3,4) = G x m_(1,4).T().D()*7" << endl;
    }
    double t4_start = MPI_Wtime();
    diagonalizeV(m_14, dm, Column, 7);
    multDimApplyPrune(m_34, dm, Column, true);
    m_34.PrintInfo();
    double t4_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 5 (Total) : " << (t4_end - t4_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 5
    if (myrank == 0) {
        cout << "step 5 : m_(2,3) = G.T() x m_(3,4).D()*17" << endl;
    }
    double t5_start = MPI_Wtime();
    diagonalizeV(m_34, dm, Row, 17);
    multDimApplyPrune(m_23, dm, Column, true);
    m_23.PrintInfo();
    double t5_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 3 (Total) : " << (t5_end - t5_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 6
    if (myrank == 0) {
        cout << "step 6 : m_(2,3) = {1@(291959481,291959481)} x m_(2,3)" << endl;
    }
    double t6_start = MPI_Wtime();
    multDimApplyPrune(m_23, l_23, Row, false);
    m_23.PrintInfo();
    double t6_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 6 (Total) : " << (t6_end - t6_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 7
    if (myrank == 0) {
        cout << "step 7 : m_(3,4) = m_(2,3).T().D() x m_(3,4)" << endl;
    }
    double t7_start = MPI_Wtime();
    diagonalizeV(m_23, dm, Column);
    multDimApplyPrune(m_34, dm, Row, true);
    m_34.PrintInfo();
    double t7_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 7 (Total) : " << (t7_end - t7_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 8
    if (myrank == 0) {
        cout << "step 8 : m_(4,0) = m_(3,4).T().D() x m_(4,0)" << endl;
    }
    double t8_start = MPI_Wtime();
    diagonalizeV(m_34, dm, Column);
    multDimApplyPrune(m_40, dm, Row, true);
    m_40.PrintInfo();
    double t8_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 8 (Total) : " << (t8_end - t8_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    double query_counting = MPI_Wtime();

    if (myrank == 0) {
        cout << "query6 mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "query6 prune time : " << total_prune_time << " s" << endl;
        cout << "query6 diag_reduce time : " << total_reduce_time << " s" << endl;
        cout << "query6 dim_apply time : " << total_dim_apply_time << " s" << endl;
        cout << "query6 total query execution time : " << query_counting - total_computing_1 << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    double resgen_start = MPI_Wtime();
    resgen_l6(m_40, m_14, m_34, m_23);
    double resgen_end = MPI_Wtime();

    // end count time
    double total_computing_2 = MPI_Wtime();
    if (myrank == 0) {
        cout << "query6 result_enum time : " << resgen_end - resgen_start << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "query6 time (Total) : " << total_computing_2 - total_computing_1 << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }
}

void resgen_l7(PSpMat::MPI_DCCols &m_30, PSpMat::MPI_DCCols &m_43, PSpMat::MPI_DCCols &m_14, PSpMat::MPI_DCCols &m_54,
               PSpMat::MPI_DCCols &m_25, PSpMat::MPI_DCCols &m_65) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t1_start = MPI_Wtime();

    clear_result_time();

    auto commGrid = m_30.getcommgrid();

    // m_30 becoms m_03
    m_30.Transpose();

    vector<IndexType> indl, indr, indj;

    vector<IndexType> order1 = {0, 0, 0, 1, 1, 0}, order2 = {0, 0, 0, 1, 0, 2, 1, 0}, order3 = {0, 0, 0, 1, 0, 2, 0, 3},
            order4 = {0, 0, 1, 0, 0, 1, 0, 2, 0, 3}, order5 = {0, 0, 1, 0, 0, 1, 0, 2, 0, 3, 0, 4};

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

    get_local_indices(m_25, indr);
    send_local_indices(commGrid, indr);
    m_25.FreeMemory();

    local_join(commGrid, indl, indr, 4, 2, 3, 1, order4, indj);
    local_redistribution(m_14, indj, 5, 3, indl);

    get_local_indices(m_14, indr);
    send_local_indices(commGrid, indr);
    m_14.FreeMemory();

    local_join(commGrid, indl, indr, 5, 2, 3, 1, order5, indj);
    send_local_results(commGrid, indj.size() / 6);
}

void lubm_l7(PSpMat::MPI_DCCols &G, PSpMat::MPI_DCCols &tG, FullyDistVec<IndexType, IndexType> &nonisov) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    clear_query_time();

    auto commWorld = G.getcommgrid();

    FullyDistVec<IndexType, ElementType> dm(commWorld);

    auto m_30(G), m_43(tG), m_14(tG), m_54(G), m_25(tG), m_65(G);

    IndexType ind1 = nonisov.FindInds(std::bind2nd(std::equal_to<IndexType>(), static_cast<IndexType>(223452631)))[0];
    IndexType ind2 = nonisov.FindInds(std::bind2nd(std::equal_to<IndexType>(), static_cast<IndexType>(235928023)))[0];
    IndexType ind3 = nonisov.FindInds(std::bind2nd(std::equal_to<IndexType>(), static_cast<IndexType>(291959481)))[0];

    FullyDistVec<IndexType, ElementType> r_30(commWorld, G.getnrow(), 0), l_14(commWorld, G.getnrow(), 0), l_25(
            commWorld, G.getnrow(), 0);
    r_30.SetElement(ind1, 17);
    l_14.SetElement(ind2, 1);
    l_25.SetElement(ind3, 1);

    // start count time
    double total_computing_1 = MPI_Wtime();

    // ==> step 1
    if (myrank == 0) {
        cout << "\n###############################################################" << endl;
        cout << "Query 7" << endl;
        cout << "###############################################################" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "step 1 : m_(3,0) = G x {1@(223452631,223452631)}*17" << endl;
    }
    double t1_start = MPI_Wtime();
    multDimApplyPrune(m_30, r_30, Column, true);
    m_30.PrintInfo();
    double t1_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 1 (Total) : " << (t1_end - t1_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 2
    if (myrank == 0) {
        cout << "step 2 : m_(4,3) = G.T() x m_(3,0).D()*4" << endl;
    }
    double t2_start = MPI_Wtime();
    diagonalizeV(m_30, dm, Row, 4);
    multDimApplyPrune(m_43, dm, Column, true);
    m_43.PrintInfo();
    double t2_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 2 (Total) : " << (t2_end - t2_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 3
    if (myrank == 0) {
        cout << "step 3 : m_(1,4) = G.T() x m_(4,3).D()*17" << endl;
    }
    double t3_start = MPI_Wtime();
    diagonalizeV(m_43, dm, Row, 17);
    multDimApplyPrune(m_14, dm, Column, true);
    m_14.PrintInfo();
    double t3_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 3 (Total) : " << (t3_end - t3_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 4
    if (myrank == 0) {
        cout << "step 4 : m_(1,4) = {1@(235928023,235928023)} x m_(1,4)" << endl;
    }
    double t4_start = MPI_Wtime();
    multDimApplyPrune(m_14, l_14, Row, false);
    m_14.PrintInfo();
    double t4_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 4 (Total) : " << (t4_end - t4_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 5
    if (myrank == 0) {
        cout << "step 5 : m_(5,4) = G x m_(1,4).T().D()*5" << endl;
    }
    double t5_start = MPI_Wtime();
    diagonalizeV(m_14, dm, Column, 5);
    multDimApplyPrune(m_54, dm, Column, true);
    m_54.PrintInfo();
    double t5_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 5 (Total) : " << (t5_end - t5_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 6
    if (myrank == 0) {
        cout << "step 6 : m_(2,5) = G.T() x m_(5,4).D()*17" << endl;
    }
    double t6_start = MPI_Wtime();
    diagonalizeV(m_54, dm, Row, 17);
    multDimApplyPrune(m_25, dm, Column, true);
    m_25.PrintInfo();
    double t6_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 6 (Total) : " << (t6_end - t6_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 7
    if (myrank == 0) {
        cout << "step 7 : m_(2,5) = {1@(291959481,291959481)} x m_(2,5)" << endl;
    }
    double t7_start = MPI_Wtime();
    multDimApplyPrune(m_25, l_25, Row, false);
    m_25.PrintInfo();
    double t7_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 7 (Total) : " << (t7_end - t7_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    // ==> step 8
    if (myrank == 0) {
        cout << "step 8 : m_(6,5) = G x m_(2,5).T().D()*18" << endl;
    }
    double t8_start = MPI_Wtime();
    diagonalizeV(m_25, dm, Column, 18);
    multDimApplyPrune(m_65, dm, Column, true);
    m_65.PrintInfo();
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
    m_65.PrintInfo();
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
    m_43.PrintInfo();
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
    m_54.PrintInfo();
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
    m_43.PrintInfo();
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
    m_30.PrintInfo();
    double t13_end = MPI_Wtime();

    if (myrank == 0) {
        cout << "step 13 (Total) : " << (t13_end - t13_start) << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    double query_counting = MPI_Wtime();

    if (myrank == 0) {
        cout << "query7 mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "query7 prune time : " << total_prune_time << " s" << endl;
        cout << "query7 diag_reduce time : " << total_reduce_time << " s" << endl;
        cout << "query7 dim_apply time : " << total_dim_apply_time << " s" << endl;
        cout << "query7 total query execution time : " << query_counting - total_computing_1 << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
    }

    double resgen_start = MPI_Wtime();
    resgen_l7(m_30, m_43, m_14, m_54, m_25, m_65);
    double resgen_end = MPI_Wtime();

    // end count time
    double total_computing_2 = MPI_Wtime();
    if (myrank == 0) {
        cout << "query7 result_enum time : " << resgen_end - resgen_start << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "query7 time (Total) : " << total_computing_2 - total_computing_1 << " s" << endl;
        cout << "---------------------------------------------------------------" << endl;
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

    if (argc < 2) {
        if (myrank == 0) {
            cout << "Usage: ./lubm_scal file" << endl;
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
            cout << "starting reading lubm data......" << endl;
        }

        PSpMat::MPI_DCCols G(MPI_COMM_WORLD);
        auto commWorld = G.getcommgrid();

        string Mname(argv[1]);

        double t1 = MPI_Wtime();
        G.ParallelReadMM(Mname, true, selectSecond);
        double t2 = MPI_Wtime();

        G.PrintInfo();
        float imG = G.LoadImbalance();

        if (myrank == 0) {
            cout << "\tread file takes : " << (t2 - t1) << " s" << endl;
            cout << "\toriginal imbalance of G : " << imG << endl;
        }

        FullyDistVec<IndexType, IndexType> nonisov(commWorld);
        permute(G, nonisov);

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
            try {    lubm_l1(G, tG, nonisov);    }
            catch (...) {    if (myrank == 0) {cout << "query 1 failed at iteration " << time << endl; }     }
            try {    lubm_l2(G, tG, nonisov);    }
            catch (...) {    if (myrank == 0) {cout << "query 2 failed at iteration " << time << endl; }     }
            try {    lubm_l3(G, tG, nonisov);    }
            catch (...) {    if (myrank == 0) {cout << "query 3 failed at iteration " << time << endl; }     }
            try {    lubm_l4(G, tG, nonisov);    }
            catch (...) {    if (myrank == 0) {cout << "query 4 failed at iteration " << time << endl; }     }
            try {    lubm_l5(G, tG, nonisov);    }
            catch (...) {    if (myrank == 0) {cout << "query 5 failed at iteration " << time << endl; }     }
            try {    lubm_l6(G, tG, nonisov);    }
            catch (...) {    if (myrank == 0) {cout << "query 6 failed at iteration " << time << endl; }     }
            try {    lubm_l7(G, tG, nonisov);    }
            catch (...) {    if (myrank == 0) {cout << "query 7 failed at iteration " << time << endl; }     }
        }
    }

    MPI_Finalize();
    return 0;
}
