#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "../include/CombBLAS.h"

using namespace std;
using namespace combblas;

template<class NT>
class PSpMat {
public:
    typedef SpDCCols<int, NT> DCCols;
    typedef SpParMat<int, NT, DCCols> MPI_DCCols;
};

#define ElementType int

bool isZero(ElementType t) {
    return t == 0;
}

int main(int argc, char *argv[]) {
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 1) {
        if (myrank == 0) {
            cout << "Usage: ./SemiringMult" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    {
        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));

        if (myrank == 0) {
            cout << "\none matrix is from file, the other is from construction" << endl;
        }

        string A1name("/home/cheny0l/work/db245/CombBLAS_beta_16_1/TESTDATA/gen_2_2_3.txt");

        MPI_Barrier(MPI_COMM_WORLD);

        typedef RDFRing<ElementType, ElementType> RDFINTINT;
        typedef PlusTimesSRing<ElementType, ElementType> PTINITINT;

        PSpMat<ElementType>::MPI_DCCols A1(MPI_COMM_WORLD);

        A1.ReadDistribute(A1name, 0);
        A1.PrintInfo();

        int nrow1 = A1.getnrow(), ncol1 = A1.getncol();
        int ris[3] = {0, 0, 1};
        std::vector<int> riv(ris, ris + sizeof(ris) / sizeof(int));

        int cis[3] = {0, 1, 1};
        std::vector<int> civ(cis, cis + sizeof(cis) / sizeof(int));

        int vis[3] = {2, 1, 2};
        std::vector<int> viv(vis, vis + sizeof(vis) / sizeof(int));

        FullyDistVec<int, ElementType> ri(riv, fullWorld);
        if (myrank == 0) {
            std::cout << "ri : ";
        }
        ri.DebugPrint();
        FullyDistVec<int, ElementType> ci(civ, fullWorld);
        if (myrank == 0) {
            std::cout << "ci : ";
        }
        ci.DebugPrint();
        FullyDistVec<int, ElementType> vi(viv, fullWorld);
        if (myrank == 0) {
            std::cout << "vi : ";
        }
        vi.DebugPrint();

        PSpMat<ElementType>::MPI_DCCols B1(nrow1, ncol1, ri, ci, vi);

        B1.PrintInfo();

        double t1 = MPI_Wtime();
        PSpMat<ElementType>::MPI_DCCols C11 = Mult_AnXBn_DoubleBuff<PTINITINT, ElementType, PSpMat<ElementType>::DCCols>(A1, B1);
        PSpMat<ElementType>::MPI_DCCols C12 = Mult_AnXBn_DoubleBuff<RDFINTINT, ElementType, PSpMat<ElementType>::DCCols>(A1, B1);
        double t2 = MPI_Wtime();
        if(myrank == 0) {
            cout << "multiplication takes " << t2 - t1 << " s" << endl;
        }

        if (myrank == 0) {
            cout << "normal multiplication : \n";
        }
        C11.PrintInfo();
        C11.Prune(isZero);
        C11.PrintInfo();
        C11.SaveGathered("./C11-normal-construct.txt");

        if (myrank == 0) {
            cout << "semiring multiplication : \n";
        }
        C12.PrintInfo();
        C12.Prune(isZero);
        C12.PrintInfo();
        C12.SaveGathered("./C12-semiring-construct.txt");

        if (myrank == 0) {
            cout << "\ntwo matrixes both from file" << endl;
        }

        string Aname2("/home/cheny0l/work/db245/CombBLAS_beta_16_1/TESTDATA/gen_2_2_3.txt");
        string Bname2("/home/cheny0l/work/db245/CombBLAS_beta_16_1/TESTDATA/gen_2_2_B.txt");

        MPI_Barrier(MPI_COMM_WORLD);

        typedef RDFRing<ElementType, ElementType> RDFINTINT;
        PSpMat<ElementType>::MPI_DCCols A2(MPI_COMM_WORLD), B2(MPI_COMM_WORLD);

        A2.ReadDistribute(Aname2, 0);
        A2.PrintInfo();

        B2.ReadDistribute(Bname2, 0);
        B2.PrintInfo();

        double t3 = MPI_Wtime();
        PSpMat<ElementType>::MPI_DCCols C21 = Mult_AnXBn_DoubleBuff<PTINITINT, ElementType, PSpMat<ElementType>::DCCols>(A1, B1);
        PSpMat<ElementType>::MPI_DCCols C22 = Mult_AnXBn_DoubleBuff<RDFINTINT, ElementType, PSpMat<ElementType>::DCCols>(A2, B2);
        double t4 = MPI_Wtime();
        if(myrank == 0) {
            cout << "multiplication takes " << t4 - t3 << " s" << endl;
        }

        if (myrank == 0) {
            cout << "normal multiplication : \n";
        }
        C21.PrintInfo();
        C21.Prune(isZero);
        C21.PrintInfo();
        C21.SaveGathered("./C21-normal-file.txt");

        if (myrank == 0) {
            cout << "semiring multiplication : \n";
        }
        C22.PrintInfo();
        C22.Prune(isZero);
        C22.PrintInfo();
        C22.SaveGathered("./C22-semiring-file.txt");
    }

    MPI_Finalize();
    return 0;
}
