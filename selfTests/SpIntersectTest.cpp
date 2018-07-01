#include <iostream>
#include <functional>
#include <algorithm>
#include <sstream>
#include "../include/CombBLAS.h"

#define IndexType int
#define ElementType int

using namespace std;
using namespace combblas;

class PSpMat {
public:
    typedef SpDCCols<IndexType, ElementType> DCCols;
    typedef SpParMat<IndexType, ElementType, DCCols> MPI_DCCols;
};

int main(int argc, char *argv[]) {
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 3) {
        if (myrank == 0) {
            cout << "Usage: ./SpRefTest <MatrixA> <MatrixB>" << endl;
            cout << "<MatrixA>, <MatrixB> are file addresses, and file should be in triples format" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    {
        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

        string Aname(argv[1]), Bname(argv[2]);

        MPI_Barrier(MPI_COMM_WORLD);

        PSpMat::MPI_DCCols A(MPI_COMM_WORLD), B(MPI_COMM_WORLD);

        double t1 = MPI_Wtime();
        A.ReadDistribute(Aname, 0);
//        A.ParallelReadMM(Mname, true, maximum<ElementType>());
        double t2 = MPI_Wtime();
        if(myrank == 0) {
            cout << "read file takes " << t2 - t1 << " s" << endl;
        }
        A.PrintInfo();

        double t3 = MPI_Wtime();
        B.ReadDistribute(Bname, 0);
//        B.ParallelReadMM(Mname, true, maximum<ElementType>());
        double t4 = MPI_Wtime();
        if(myrank == 0) {
            cout << "read file takes " << t4 - t3 << " s" << endl;
        }
        B.PrintInfo();

        auto dA = A.seq().GetInternal();
        auto dB = B.seq().GetInternal();

        if (myrank == 0) {
            cout << "nz of A : " << dA->nz << endl;
            cout << "nz of B : " << dB->nz << endl;
        }

        Isect<IndexType> *isect1, *isect2, *itr1, *itr2, *cols, *rows;
        SpHelper::SpIntersect(*dA, *dB, cols, rows, isect1, isect2, itr1, itr2);

        if (myrank == 0) {
            // condition : A.row == B.row && A.col == B.col
            // not very useful for me
            cout << "size of intersection : " << itr1 - isect1 << endl;
        }
    }

    MPI_Finalize();
    return 0;
}
