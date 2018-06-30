#include <iostream>
#include <functional>
#include <algorithm>
#include <sstream>
#include <cmath>
#include "../include/CombBLAS.h"

using namespace std;
using namespace combblas;

#define IndexType int
#define ElementType int

class PSpMat {
public:
    typedef SpDCCols<IndexType, ElementType> DCCols;
    typedef SpParMat<IndexType, ElementType, DCCols> MPI_DCCols;
};

// M should have same rows and cols
// fill I and J, they should have same size
void get_indices_local(PSpMat::MPI_DCCols &M, vector<IndexType> &I, vector<IndexType> &J, vector<ElementType> &V) {
    assert(M.getnrow() = M.getncol());

    int nproc, myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    int grid_size = int(sqrt(nproc));
    int grid_length = M.getncol() / grid_size;

//    if (myrank == 0) {
//        cout << "grid_length : " << grid_length << endl;
//    }

    auto commGrid = M.getcommgrid();

    auto d0 = M.seq().GetInternal();

    int colrank = commGrid->GetRankInProcCol();
    int rowrank = commGrid->GetRankInProcRow();

    I.assign(d0->ir, d0->ir + d0->nz);
    transform(I.begin(), I.end(), I.begin(), bind2nd(std::plus<int>(), colrank * grid_length));
    V.assign(d0->numx, d0->numx+d0->nz);

    for (int index = 0; index < d0->nzc; ++index) {
        int times = d0->cp[index + 1] - d0->cp[index];

        for (int i = 0; i < times; ++i) {
            J.push_back(d0->jc[index]);
        }
    }
    transform(J.begin(), J.end(), J.begin(), bind2nd(std::plus<int>(), rowrank * grid_length));

    // if does not have same size, wrong
    assert(I.size() == J.size());
}

int main(int argc, char *argv[]) {
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 2) {
        if (myrank == 0) {
            cout << "Usage: ./getdcsc <Matrix>" << endl;
            cout << "<Matrix> is file address, and file should be in triples format" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    {
        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));

        string Mname(argv[1]);

        MPI_Barrier(MPI_COMM_WORLD);

        PSpMat::MPI_DCCols A(MPI_COMM_WORLD);

        double t1 = MPI_Wtime();
        A.ReadDistribute(Mname, 0);
//        A.ParallelReadMM(Mname, true, maximum<ElementType>());
        double t2 = MPI_Wtime();
        if (myrank == 0) {
            cout << "read file takes " << t2 - t1 << " s" << endl;
        }
//        A.PrintInfo();

        // get I and J
//        auto d0 = A.seq().GetInternal();
////
////        cout << "myrank : " << myrank << "\t";
////        cout << "nz : " << d0->nz << "\t";
////        cout << "nzc : " << d0->nzc << endl;
////
        vector<IndexType> I, J, V;
//        for (int index = 0; index < d0->nzc; ++index) {
//            int times = d0->cp[index + 1] - d0->cp[index];
//
//            for (int i = 0; i < times; ++i) {
//                J.push_back(d0->jc[index]);
//            }
//        }
//
//        cout << myrank << "\t" << I.size() << "\t" << J.size() << endl;

        get_indices_local(A, I, J, V);

//        cout << myrank << "\t" << I.size() << "\t" << J.size() << endl;
        cout << "myrank : " << myrank << "\t";
        for (int i = 0; i < I.size(); ++i) {
            cout << I[i]+1 << "," << J[i]+1 << "," << V[i] << "\t";
        }
        cout << endl;

    }

    MPI_Finalize();
    return 0;
}
