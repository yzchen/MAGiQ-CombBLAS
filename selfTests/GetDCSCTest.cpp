#include <iostream>
#include <functional>
#include <algorithm>
#include <sstream>
#include <fstream>
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
void get_indices_local(PSpMat::MPI_DCCols &M, vector<IndexType> &I, vector<IndexType> &J) {
    assert(M.getnrow() == M.getncol());

    auto commGrid = M.getcommgrid();
    int colrank = commGrid->GetRankInProcCol();
    int rowrank = commGrid->GetRankInProcRow();

    int nproc, myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

//    int grid_size = int(sqrt(nproc));
//    int grid_length = M.getncol() / grid_size;

    int colneighs = commGrid->GetGridRows();
    IndexType *locnrows = new IndexType[colneighs];  // number of rows is calculated by a reduction among the processor column
    locnrows[colrank] = M.getlocalrows();
    MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IndexType>(), locnrows, 1, MPIType<IndexType>(), commGrid->GetColWorld());
    IndexType roffset = std::accumulate(locnrows, locnrows + colrank, 0);
    delete[] locnrows;

    int rowneighs = commGrid->GetGridCols();
    IndexType *locncols = new IndexType[rowneighs];  // number of rows is calculated by a reduction among the processor column
    locncols[rowrank] = M.getlocalcols();
    MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IndexType>(), locncols, 1, MPIType<IndexType>(), commGrid->GetRowWorld());
    IndexType coffset = std::accumulate(locncols, locncols + rowrank, 0);
    delete[] locncols;

//    if (myrank == 0) {
//        cout << "grid_length : " << grid_length << endl;
//    }

    //// if there is nothing in current process, then d0 will be NULL pointer
    auto d0 = M.seq().GetInternal();

//    cout << myrank << ", " << (d0 == NULL) << endl;

//    cout << myrank << ", nz = " << d0->nz << endl;

    cout << "offset of process " << myrank << ", roffset = " << roffset << ", coffset = " << coffset << endl;

    if (d0 != NULL) {
        double t1 = MPI_Wtime();
        I.assign(d0->ir, d0->ir + d0->nz);
        transform(I.begin(), I.end(), I.begin(), bind2nd(std::plus<int>(), roffset));
        double t2 = MPI_Wtime();
        if (myrank == 0) {
            cout << myrank << ", construct I takes : " << (t2 - t1) << " s" << endl;
        }

        double t5 = MPI_Wtime();
        for (int index = 0; index < d0->nzc; ++index) {
            int times = d0->cp[index + 1] - d0->cp[index];

            for (int i = 0; i < times; ++i) {
                J.push_back(d0->jc[index]);
            }
        }
        transform(J.begin(), J.end(), J.begin(), bind2nd(std::plus<int>(), coffset));
        double t6 = MPI_Wtime();
        if (myrank == 0) {
            cout << myrank << ", construct J takes : " << (t6 - t5) << " s" << endl;
        }

        // if does not have same size, wrong
        assert(I.size() == J.size());


        // output to file
        stringstream os;
        os << "dcsc/" << myrank << ".txt";

        double t7 = MPI_Wtime();
        std::ofstream outFile(os.str());
        for (int i = 0; i < I.size(); i++)
            outFile << I[i] + 1 << "\t" << J[i] + 1 << "\n";
        double t8 = MPI_Wtime();
        cout << "output indices results for process " << myrank << " takes : " << (t8 - t7) << " s" << endl;
    }

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
//        A.ReadDistribute(Mname, 0);
        A.ParallelReadMM(Mname, true, maximum<ElementType>());
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
        vector<IndexType> I, J;
//        for (int index = 0; index < d0->nzc; ++index) {
//            int times = d0->cp[index + 1] - d0->cp[index];
//
//            for (int i = 0; i < times; ++i) {
//                J.push_back(d0->jc[index]);
//            }
//        }
//
//        cout << myrank << "\t" << I.size() << "\t" << J.size() << endl;

        get_indices_local(A, I, J);

//        cout << myrank << "\t" << I.size() << "\t" << J.size() << endl;
//        cout << "myrank : " << myrank << "\t";
//        for (int i = 0; i < I.size(); ++i) {
//            cout << I[i]+1 << "," << J[i]+1 << "," << V[i] << "\t";
//        }
//        cout << endl;

    }

    MPI_Finalize();
    return 0;
}
