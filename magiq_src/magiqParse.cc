#include <iostream>
#include <functional>
#include <algorithm>
#include <sstream>
#include <vector>
#include <iterator>
#include <fstream>
#include "../magiq_include/magiqScal.h"

using namespace std;
using namespace combblas;

int main(int argc, char *argv[]) {
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // initialize comp function array
    initComp();

    if (argc < 3) {
        if (myrank == 0) {
            cout << "Usage: ./magiqParse dataFile sparqlFile isPerm=false repeat=1" << endl << flush;
        }
        MPI_Finalize();
        return -1;
    }

    {
        // initialization phase
        MPI_Barrier(MPI_COMM_WORLD);
        string dataName(argv[1]), sparqlFile(argv[2]);
        int isPerm = 0;
        if (argc > 3)
            isPerm = atoi(argv[3]);
        
        if (myrank == 0) {
            cout << "###############################################################" << endl << flush;
            cout << "Load Matrix" << endl << flush;
            cout << "###############################################################" << endl << flush;
            cout << "---------------------------------------------------------------" << endl << flush;
            cout << "All procs reading and permuting input graph [" << dataName << "]..." << endl << flush;
        }

        PSpMat::MPI_DCCols G(MPI_COMM_WORLD);
        auto commWorld = G.getcommgrid();

        // permute vector 
        FullyDistVec<IndexType, IndexType> nonisov(commWorld);

        double t1 = MPI_Wtime();
        G.ParallelReadMM(dataName, true, selectSecond, isPerm > 0, nonisov);
        double t2 = MPI_Wtime();

        if (myrank == 0) {
            cout << "---------------------------------------------------------------" << endl << flush << flush;;
        }
        G.PrintInfo();
        float imG = G.LoadImbalance();

        if (myrank == 0) {
            cout << "\tread graph took : " << (t2 - t1) << " s" << endl << flush;
            cout << "\timbalance of G : " << imG << endl << flush;
        }
        
        double t2_trans = MPI_Wtime();

        if (myrank == 0) {
            cout << "graph load (Total) : " << (t2_trans - t1) << " s" << endl << flush;
            cout << "---------------------------------------------------------------" << endl << flush << flush;;
        }

        map<string, PSpMat::MPI_DCCols> matrices;
        FullyDistVec<IndexType, ElementType> dm(commWorld);
        
        // run the same query several(repeat) times
        int repeat = 1;
        if (argc > 4)
            repeat = atoi(argv[4]);
        for (int i = 0; i < repeat; i++) {
            parseSparql(sparqlFile.c_str(), matrices, G, dm, isPerm > 0, nonisov);
        }
    }

    MPI_Finalize();
    return 0;
}