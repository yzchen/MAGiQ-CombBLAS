#include <iostream>
#include <functional>
#include <algorithm>
#include <sstream>
#include <vector>
#include <iterator>
#include <fstream>
#include "../include/CombBLAS.h"

using namespace std;
using namespace combblas;

#define IndexType uint32_t
#define ElementType uint8_t

class PSpMat {
public:
    typedef SpDCCols<IndexType, ElementType> DCCols;
    typedef SpParMat<IndexType, ElementType, DCCols> MPI_DCCols;
};

ElementType selectSecond(ElementType a, ElementType b) {    return b;   }

int main(int argc, char *argv[]) {
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    
    if (argc < 1) {
        if (myrank == 0) {
            cout << "Usage: ./lubm10241" << endl;
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

        // string Mname("/home/cheny0l/work/db245/fuad/data/lubm10240/encoded.mm");
       string Mname("/scratch/cheny0l/lubm_small/data/lubm10240/encoded.mm");

        double t1 = MPI_Wtime();
        PSpMat::MPI_DCCols G(MPI_COMM_WORLD);
        G.ParallelReadMM(Mname, true, selectSecond);
        double t2 = MPI_Wtime();

        G.PrintInfo();
        auto commWorld = G.getcommgrid();
        float imG = G.LoadImbalance();

        if (myrank == 0) {
            cout << "\tread file takes : " << (t2 - t1) << " s" << endl;
            cout << "\toriginal imbalance of G : " << imG << endl;
        }
    }

    MPI_Finalize();
    return 0;
}
