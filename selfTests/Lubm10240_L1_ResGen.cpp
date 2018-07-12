#include <iostream>
#include <functional>
#include <algorithm>
#include <sstream>
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

    if (myrank == 0) {
        cout << "---------------------------------------------------------------" << endl;
        cout << "begin result generation ......" << endl;
    }

    auto commGrid = m_50.getcommgrid();

    // m_50 becoms m_05
    m_50.Transpose();
    vector<IndexType> index_05;
    get_local_indices(m_50, index_05);
    send_local_indices(commGrid, index_05);
    write_local_vector(index_05, "m_50", 2);

    vector<IndexType> index_35;
    get_local_indices(m_35, index_35);
    send_local_indices(commGrid, index_35);
    write_local_vector(index_35, "m_35", 2);

    vector<IndexType> order1 = {0, 0, 1, 0, 1, 1};
    vector<IndexType> index_035_0, index_035;

    local_join(commGrid, index_05, index_35, 2, 2, 1, 1, order1, index_035_0);

    write_local_vector(index_035_0, "index_035_0", 3);
    local_redistribution(m_43, index_035_0, 3, 1, index_035);
    index_035_0.clear();

    vector<IndexType> index_43;
    get_local_indices(m_43, index_43);
    send_local_indices(commGrid, index_43);
    write_local_vector(index_43, "m_43", 2);

    vector<IndexType> order2 = {0, 0, 0, 1, 1, 0, 0, 2};
    vector<IndexType> index_0345_0, index_0345;

    local_join(commGrid, index_035, index_43, 3, 2, 1, 1, order2, index_0345_0);

    local_redistribution(m_64, index_0345_0, 4, 2, index_0345);
    write_local_vector(index_0345, "index_0345", 4);
    index_0345_0.clear();

    vector<IndexType> index_64;
    get_local_indices(m_64, index_64);
    send_local_indices(commGrid, index_64);
    write_local_vector(index_64, "m_64", 2);

    vector<IndexType> order3 = {0, 0, 0, 1, 0, 2, 0, 3};
    vector<IndexType> index_03456;

//    local_join(commGrid, index_0345, index_64, 4, 2, 2, 1, order3, index_03456);
    local_filter(commGrid, index_0345, index_64, 4, 2, 2, 3, 1, 0, order3, index_03456);

    write_local_vector(index_03456, "index_03456", 4);

    vector<IndexType> index_24;
    get_local_indices(m_24, index_24);
    send_local_indices(commGrid, index_24);
    write_local_vector(index_24, "m_24", 2);

    vector<IndexType> order4 = {0, 0, 1, 0, 0, 1, 0, 2, 0, 3};
    vector<IndexType> index_023456_0, index_023456;

    local_join(commGrid, index_03456, index_24, 4, 2, 2, 1, order4, index_023456_0);

    write_local_vector(index_023456_0, "index_023456_0", 5);


    local_redistribution(m_13, index_023456_0, 5, 2, index_023456);
    index_023456_0.clear();

    // TODO : code check point
    cout << myrank << " check point, finished 3 joins and 1 filter before 1111111" << endl;

    vector<IndexType> index_13;
    get_local_indices(m_13, index_13);
    write_local_vector(index_13, "m_13", 2);
    send_local_indices(commGrid, index_13);

    vector<IndexType> order5 = {0, 0, 1, 0, 0, 1, 0, 2, 0, 3, 0, 4};
    vector<IndexType> index_0123456;

    // TODO : code check point
    cout << myrank << " check point, finished 3 joins and 1 filter " << endl;

    local_join(commGrid, index_023456, index_13, 5, 2, 2, 1, order5, index_0123456);


    send_local_results(commGrid, index_0123456.size() / 6);
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

    // end count time
    double total_computing_2 = MPI_Wtime();

    printReducedInfo(m_50);

    if (myrank == 0) {
        cout << "query1 mmul_scalar time : " << total_mmul_scalar_time << " s" << endl;
        cout << "query1 prune time : " << total_prune_time << " s" << endl;
        cout << "query1 diag_reduce time : " << total_reduce_time << " s" << endl;
        cout << "query1 dim_apply time : " << total_dim_apply_time << " s" << endl;
        cout << "query1 time (Total) : " << total_computing_2 - total_computing_1 << " s" << endl;
    }

    resgen_l1(m_50, m_35, m_43, m_64, m_24, m_13);

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

        lubm10240_l1(G, tG, nonisov);


    }

    MPI_Finalize();
    return 0;
}
