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

    if (argc < 2) {
        if (myrank == 0) {
            cout << "Usage: ./sendrecv <MatrixA>" << endl;
            cout << "<MatrixA> is file addresses, and file should be in triples format" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    {
        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));

        string Aname(argv[1]);

        MPI_Barrier(MPI_COMM_WORLD);

        PSpMat::MPI_DCCols A(MPI_COMM_WORLD);

        double t1 = MPI_Wtime();
        A.ReadDistribute(Aname, 0);
//        A.ParallelReadMM(Mname, true, maximum<ElementType>());
        double t2 = MPI_Wtime();
        if (myrank == 0) {
            cout << "read file takes " << t2 - t1 << " s" << endl;
        }
        A.PrintInfo();

        auto commGrid = A.getcommgrid();
        IndexType n_thisrow = A.getlocalrows();

        std::vector<ElementType> sendbuf(n_thisrow, myrank);
        std::vector<IndexType> send_rowdisp(n_thisrow + 1, myrank);
        std::vector<IndexType> local_rowdisp(n_thisrow + 1, myrank);

        std::vector<ElementType> recvbuf(n_thisrow);
        std::vector<ElementType> tempbuf(n_thisrow);
        std::vector<IndexType> recv_rowdisp(n_thisrow + 1);
        std::vector<IndexType> templen(n_thisrow);

        int rowneighs = commGrid->GetGridCols();
        int rowrank = commGrid->GetRankInProcRow();

        // prepare data

        for (int p = 2; p <= rowneighs; p *= 2) {

            if (rowrank % p == p / 2) { // this processor is a sender in this round
                int receiver = rowrank - ceil(p / 2);
                MPI_Send(send_rowdisp.data(), n_thisrow + 1, MPIType<IndexType>(), receiver, 0,
                         commGrid->GetRowWorld());
                MPI_Send(sendbuf.data(), send_rowdisp[n_thisrow], MPIType<ElementType>(), receiver, 1,
                         commGrid->GetRowWorld());
                //break;
                cout << "round " << p/2 << ", " << myrank << " sender" << endl;
            } else if (rowrank % p == 0) { // this processor is a receiver in this round
                int sender = rowrank + ceil(p / 2);
                if (sender < rowneighs) {

                    MPI_Recv(recv_rowdisp.data(), n_thisrow + 1, MPIType<IndexType>(), sender, 0,
                             commGrid->GetRowWorld(), MPI_STATUS_IGNORE);
                    MPI_Recv(recvbuf.data(), recv_rowdisp[n_thisrow], MPIType<ElementType>(), sender, 1,
                             commGrid->GetRowWorld(), MPI_STATUS_IGNORE);

                    // do something
                    cout << "round " << p/2 << ", " << myrank << " receiver" << endl;
                }
            }
        }

    }

    MPI_Finalize();
    return 0;
}
