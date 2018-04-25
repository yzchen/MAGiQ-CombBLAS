#include <iostream>
#include <functional>
#include <algorithm>
#include <sstream>
#include "../include/CombBLAS.h"

using namespace std;
using namespace combblas;

#define ElementType int
#define IndexType int

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

    if (argc < 1) {
        if (myrank == 0) {
            cout << "Usage: ./permute" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    {
        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));

        MPI_Barrier(MPI_COMM_WORLD);

        string Mname("/home/cheny0l/work/db245/fuad/data/lubm320/encoded.mm");
        PSpMat::MPI_DCCols G(MPI_COMM_WORLD);

        double t1 = MPI_Wtime();

        G.ParallelReadMM(Mname, true, maximum<ElementType>());
        G.PrintInfo();

        double t2 = MPI_Wtime();

        float imG = G.LoadImbalance();
        if (myrank == 0) {
            cout << "read file takes : " << (t2 - t1) << " s" << endl;
            cout << "imbalance of G : " << imG << endl;
        }

        typedef PlusTimesSRing<ElementType, ElementType> PTINTINT;
//        int iterations = 10;
//
//        // mulitiplication
//        auto A(G);
//        double t3 = MPI_Wtime();
//        for (int i = 0; i < iterations; i++) {
//            Mult_AnXBn_DoubleBuff<PTINTINT, ElementType, PSpMat<ElementType>::DCCols>(G, A);
//        }
//        double t4 = MPI_Wtime();
//        if (myrank == 0) {
//            cout << "before permute, mult takes : " << (t4 - t3) / iterations << endl;
//        }

        // permute
        FullyDistVec<IndexType, ElementType> * ColSums = new FullyDistVec<IndexType, ElementType>(G.getcommgrid());
        FullyDistVec<IndexType, ElementType> * RowSums = new FullyDistVec<IndexType, ElementType>(G.getcommgrid());
        G.Reduce(*ColSums, Column, plus<IndexType>(), static_cast<IndexType>(0));
        G.Reduce(*RowSums, Row, plus<IndexType>(), static_cast<IndexType>(0));
        ColSums->EWiseApply(*RowSums, plus<IndexType>());

        FullyDistVec<IndexType, ElementType> nonisov = ColSums->FindInds(bind2nd(greater<IndexType>(), 0));

        nonisov.RandPerm();

        G(nonisov, nonisov, true);

        float nimG = G.LoadImbalance();
        if (myrank == 0) {
            cout << "new imbalance : " << nimG << endl;
        }

        int nrow = G.getnrow(), ncol = G.getncol();
        std::vector<int> riv(1, 107);
        std::vector<int> civ(1, 107);
        std::vector<int> viv(1, 8);

        FullyDistVec<int, ElementType> ri(riv, G.getcommgrid());
        FullyDistVec<int, ElementType> ci(civ, G.getcommgrid());
        FullyDistVec<int, ElementType> vi(viv, G.getcommgrid());

        PSpMat::MPI_DCCols r_30(nrow, ncol, ri, ci, vi);
        r_30(nonisov, nonisov, true);

        auto resv = nonisov.FindInds(std::bind2nd(std::equal_to<ElementType>(), static_cast<ElementType>(107)));
        resv.PrintInfo("resv");

        auto ind = resv[0];
        if (myrank == 0) {
            cout << "index : " << ind << endl;
        }

        std::vector<int> riv1(1, ind);
        std::vector<int> civ1(1, ind);

        FullyDistVec<int, ElementType> ri1(riv1, G.getcommgrid());
        FullyDistVec<int, ElementType> ci1(civ1, G.getcommgrid());

        PSpMat::MPI_DCCols pr_30(nrow, ncol, ri1, ci1, vi);

        // store some matrix and vectors
        nonisov.ParallelWrite("permute_nonisov.txt", false);
        resv.ParallelWrite("permute_resv.txt", false);

        r_30.SaveGathered("permute_r30.txt");
        pr_30.SaveGathered("permute_pr30.txt");

//        auto B(G);
//        double t5 = MPI_Wtime();
//        for (int i = 0; i < iterations; i++) {
//            Mult_AnXBn_DoubleBuff<PTINTINT, ElementType, PSpMat<ElementType>::DCCols>(G, B);
//        }
//        double t6 = MPI_Wtime();
//        if (myrank == 0) {
//            cout << "after permute, mult takes : " << (t6 - t5) / iterations << endl;
//        }

   }

    MPI_Finalize();
    return 0;
}
