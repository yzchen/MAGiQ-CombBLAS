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

    //// if there is nothing in current process, then d0 will be NULL pointer
    auto d0 = M.seq().GetInternal();

//    cout << "offset of process " << myrank << ", roffset = " << roffset << ", coffset = " << coffset << endl;

    if (d0 != NULL) {
        double t1 = MPI_Wtime();
        I.assign(d0->ir, d0->ir + d0->nz);
        transform(I.begin(), I.end(), I.begin(), bind2nd(std::plus<int>(), roffset));
        double t2 = MPI_Wtime();
//        if (myrank == 0) {
//            cout << myrank << ", construct I takes : " << (t2 - t1) << " s" << endl;
//        }

        double t5 = MPI_Wtime();
        for (int index = 0; index < d0->nzc; ++index) {
            int times = d0->cp[index + 1] - d0->cp[index];

            for (int i = 0; i < times; ++i) {
                J.push_back(d0->jc[index]);
            }
        }
        transform(J.begin(), J.end(), J.begin(), bind2nd(std::plus<int>(), coffset));
        double t6 = MPI_Wtime();
//        if (myrank == 0) {
//            cout << myrank << ", construct J takes : " << (t6 - t5) << " s" << endl;
//        }

        // if does not have same size, wrong
        assert(I.size() == J.size());
    }

}

void send_local_indices(PSpMat::MPI_DCCols &A, vector<IndexType> &I, vector<IndexType> &J) {
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    auto commGrid = A.getcommgrid();

//        cout << myrank << ", " << A.getlocalnnz() << endl;

    int number_count;
    int max_count = A.getlocalcols() * A.getlocalrows();

    std::vector<IndexType> recv_I(max_count);
    std::vector<IndexType> recv_J(max_count);

    int rowneighs = commGrid->GetGridCols();
    int rowrank = commGrid->GetRankInProcRow();

    // prepare data

    for (int p = 2; p <= rowneighs; p *= 2) {

        if (rowrank % p == p / 2) { // this processor is a sender in this round
            number_count = I.size();

            int receiver = rowrank - ceil(p / 2);
            MPI_Send(I.data(), number_count, MPIType<IndexType>(), receiver, 0,
                     commGrid->GetRowWorld());
            MPI_Send(J.data(), number_count, MPIType<IndexType>(), receiver, 1,
                     commGrid->GetRowWorld());
            //break;
//                cout << "round " << p / 2 << ", " << myrank << " sender" << endl;
        } else if (rowrank % p == 0) { // this processor is a receiver in this round
            MPI_Status status;

            int sender = rowrank + ceil(p / 2);
            if (sender < rowneighs) {
                MPI_Recv(recv_I.data(), max_count, MPIType<IndexType>(), sender, 0,
                         commGrid->GetRowWorld(), &status);
                MPI_Recv(recv_J.data(), max_count, MPIType<ElementType>(), sender, 1,
                         commGrid->GetRowWorld(), MPI_STATUS_IGNORE);

                // do something
                MPI_Get_count(&status, MPI_INT, &number_count);
//                    cout << "round " << p / 2 << ", " << myrank << " receiver " << number_count << endl;

                I.insert(I.end(), recv_I.begin(), recv_I.begin() + number_count);
                J.insert(J.end(), recv_J.begin(), recv_J.begin() + number_count);

                cout << "round " << p / 2 << " rank " << myrank << " has size of I " << I.size() << " and size of J  " << J.size() << endl;
            }
        }
    }

}

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
//        A.ReadDistribute(Aname, 0);
        A.ParallelReadMM(Aname, true, maximum<ElementType>());
        double t2 = MPI_Wtime();
        if (myrank == 0) {
            cout << "read file takes " << t2 - t1 << " s" << endl;
        }
        A.PrintInfo();

        auto commGrid = A.getcommgrid();

        vector<IndexType> I, J;
        get_indices_local(A, I, J);

        send_local_indices(A, I, J);

////        cout << myrank << ", " << A.getlocalnnz() << endl;
//
//        int number_count;
//        int max_count = A.getlocalcols() * A.getlocalrows();
//
//        std::vector<IndexType> recv_I(max_count);
//        std::vector<IndexType> recv_J(max_count);
//
//        int rowneighs = commGrid->GetGridCols();
//        int rowrank = commGrid->GetRankInProcRow();
//
//        // prepare data
//
//        for (int p = 2; p <= rowneighs; p *= 2) {
//
//            if (rowrank % p == p / 2) { // this processor is a sender in this round
//                number_count = I.size();
//
//                int receiver = rowrank - ceil(p / 2);
//                MPI_Send(I.data(), number_count, MPIType<IndexType>(), receiver, 0,
//                         commGrid->GetRowWorld());
//                MPI_Send(J.data(), number_count, MPIType<IndexType>(), receiver, 1,
//                         commGrid->GetRowWorld());
//                //break;
////                cout << "round " << p / 2 << ", " << myrank << " sender" << endl;
//            } else if (rowrank % p == 0) { // this processor is a receiver in this round
//                MPI_Status status;
//
//                int sender = rowrank + ceil(p / 2);
//                if (sender < rowneighs) {
//                    MPI_Recv(recv_I.data(), max_count, MPIType<IndexType>(), sender, 0,
//                             commGrid->GetRowWorld(), &status);
//                    MPI_Recv(recv_J.data(), max_count, MPIType<ElementType>(), sender, 1,
//                             commGrid->GetRowWorld(), MPI_STATUS_IGNORE);
//
//                    // do something
//                    MPI_Get_count(&status, MPI_INT, &number_count);
////                    cout << "round " << p / 2 << ", " << myrank << " receiver " << number_count << endl;
//
//                    I.insert(I.end(), recv_I.begin(), recv_I.begin() + number_count);
//                    J.insert(J.end(), recv_J.begin(), recv_J.begin() + number_count);
//
//                    cout << "round " << p / 2 << ", " << myrank << " has " << I.size() << endl;
//                }
//            }
//        }

    }

    MPI_Finalize();
    return 0;
}
