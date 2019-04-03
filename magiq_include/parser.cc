// #ifndef _PARSER_HH_
// #define _PARSER_HH_

#include <iostream>
#include <string>
#include <algorithm>
#include "magicScal.h"

using namespace std;

// default line buffer size, 100
static const size_t lineSize = 100;

void parseLine(string &line, map<string, PSpMat::MPI_DCCols*> &matrices, 
        map<string, FullyDistVec<IndexType, ElementType> *> vectors,
        PSpMat::MPI_DCCols &G, FullyDistVec<IndexType, ElementType> &dm) {
    line.erase(remove(line.begin(), line.end(), ' '), line.end());
    line.pop_back();
    cout << line << endl;

    int eqOp = line.find('=');

    // sermiring multiplication
    bool isSermiring = true;
    // columnDiag = true : Column in diagonalizeV
    // columnDiag = false : Row in diagonalizeV
    bool columnDiag = true;
    // scale for diagonalizeV
    int scale = 1;
    // columnApply = true : Column in multDimApplyPrune
    // columnApplu = false : Row in multDimApplyPrune
    bool columnApply = true;
    
    // ⊗ : 3 chars
    // × : 2 chars
    int multOp = line.find("⊗");
    if (multOp == -1) {
        multOp = line.find("×");
        isSermiring = false;
    }

    // cout << eqOp << "\t" << multOp << endl;

    string interMat = line.substr(0, eqOp);
    string mult1 = line.substr(eqOp + 1, multOp - eqOp - 1);
    string mult2 = line.substr(multOp + 2 + isSermiring);

    cout << interMat << "\t" << mult1 << "\t" << mult2 << endl;

    // start parsing from mult1
    if (mult1[0] == 'G') {  // G or G.T
        if (mult1 == "G") {
            matrices[interMat] = new PSpMat::MPI_DCCols(G);
        } else if (mult1 == "G.T") {
            matrices[interMat] = new PSpMat::MPI_DCCols(G);
            columnApply = false;
        } else {
            // error
        }

        // parse mult2
        if (mult2[0] == 'I') {
            int dimOp = mult2.find('^');
            int scaleOp = mult2.find('*');
            int pos = atoi(mult2.substr(0, eqOp).c_str());
            if (scaleOp != std::string::npos) {
                scale *= atoi(mult2.substr(eqOp + 1, scaleOp - eqOp - 1).c_str());
            }

            FullyDistVec<IndexType, ElementType> rt(commWorld, G.getnrow(), 0);
            rt[pos] = scale;

            // parameters : matrix *, fullyvec, columnApply, sermiring
            multDimApplyPrune(matrices[interMat], rt, columnApply ? Column : Row, isSermiring);

        } else if (mult2[0] == 'm') {
            int dot1 = mult2.find('.');
            int dot2 = mult2.find('.');
            int scaleOp = mult2.find('*');

            string mat = mult2.substr(0, dot1);
            if (dot1 != std::string::npos) {
                if (dot2 != dot1 || dot2 != std::string::npos) {
                    // fst : T, snd : D
                    // string fst = mult2.substr(dot1 + 1, dot2 - dot1 - 1);
                    // string snd = mult2.substr(dot2 + 1, scaleOp != std::string::npos?scaleOp - dot2 - 1:std::string::npos);
                    // m_x_x.T.D
                } else {
                    // fst : T or D
                    string fst = mult2.substr(dot1 + 1, scaleOp != std::string::npos?scaleOp - dot1 - 1:std::string::npos);
                    columnDiag = false;
                }
            }

            if (scaleOp != std::string::npos) {
                scale *= atoi(mult2.substr(scaleOp + 1).c_str());
            }

            diagonalizeV(matrices[mat], dm, columnDiag ? Column : Row, scale);
            multDimApplyPrune(matrices[interMat], dm, columnApply ? Column : Row, isSermiring);

        } else {
            // error
        }

    } else if (mult1[0] == 'I') {   // I^xxxx*xxxx
        // mult2 should be the same as interMat
        if (interMat == mult2) {
            columnApply = false;
            isSermiring = false;
            multDimApplyPrune(matrices[interMat], , columnApply ? Column : Row, isSermiring);
        } else {
            // error
        }

    } else if (mult1[0] == 'm') {   // m_x_x, m_x_x.T, m_x_x.D*xxx or m_x_x.T.D*xxxx
        if (mult2 == interMat) {
            columnApply = false;

            int dot1 = mult1.find('.');
            int dot2 = mult1.rfind('.');

            if (dot2 == dot1 || dot2 == std::string::npos) {
                string fst = mult2.substr(dot1 + 1);
            } else {
                columnDiag = false;
            }

            string mat = mult1.substr(0, dot1);
            diagonalizeV(matrices[mat], dm, columnDiag ? Column : Row, scale);
            multDimApplyPrune(matrices[interMat], dm, columnApply ? Column : Row, isSermiring);

        } else {
            int dot1 = mult2.find('.');
            int dot2 = mult2.rfind('.');

            if (dot2 == dot1 || dot2 == std::string::npos) {
                string fst = mult2.substr(dot1 + 1);
            } else {
                columnDiag = false;
            }

            string mat = mult2.substr(0, dot1);
            diagonalizeV(matrices[mat], dm, columnDiag ? Column : Row, scale);
            multDimApplyPrune(matrices[interMat], dm, columnApply ? Column : Row, isSermiring);

        }
    } else {
        // error
    }

    // end of function call
}

int parseSparql(const char* const sparqlFile, 
        map<string, PSpMat::MPI_DCCols*> &matrices, 
        map<string, FullyDistVec<IndexType, ElementType> *> vectors,
        PSpMat::MPI_DCCols &G, FullyDistVec<IndexType, ElementType> &dm) {
    //open and get the file handle
    FILE *fh;
    fh = fopen(sparqlFile, "r");

    //check if file exists
    if (fh == NULL){
        printf("file does not exists : %s\n", sparqlFile);
        return 0;
    }

    printf("starting reading sparql file ...\n");

    // line buffer
    char* line = (char *)malloc(lineSize);

    while (fgets(line, lineSize, fh) != NULL) {
        string str(line);
        parseLine(str, matrices, vectors, G, dm);
    }

    // free momery
    free(line);

    return 0;
}

// int main(int argc, char *argv[]) {
//     char fileName[] = "../parser/data.txt";
//     parseSparql(fileName);
    
//     return 0;
// }

// #endif // _PARSER_HH_