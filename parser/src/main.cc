#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#define DEBUG

extern FILE *yyin;      // flex uses yyin as input file's pointer
extern int yylex();     // lexer.cc provides yylex()
extern int yyparse();   // parser.cc provides yyparse()

int main(int argc, char *argv[]) {
    if (argc < 2) {
            std::cout << "Usage: ./parser file" << std::endl << std::flush;
        return -1;
    }

	// open a file handle to a particular file:
	// FILE *myfile = fopen("test.txt", "r");
	FILE *myfile = fopen(argv[1], "r");
	
	// make sure it is valid:
	if (!myfile) {
		std::cout << "File not found!" << std::endl;
		return -1;
	}

	// Set flex to read from it instead of defaulting to STDIN:
	yyin = myfile;
	
	// Parse through the input:
	yyparse();
}
