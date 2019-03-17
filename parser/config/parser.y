%{
	#include <cstdio>
	#include <iostream>
	#include <string>
	using namespace std;

	// stuff from flex that bison needs to know about:
	extern int yylex();
	extern int yyparse();
	extern FILE *yyin;
 
	void yyerror(const char *s);
%}

%union {
	int ival;
	std::string *sval;
}

%token <ival> INT_TOK UNDERLINE_TOK ASSIGN_TOK DIM_TOK DOT_TOK SCALAR_TOK NORMAL_MATMUL_TOK SERMIRING_MATMUL_TOK ERROR_TOK
%token <sval> STRING_TOK

%%

QueryUnit:
	QueryUnit QueryStatement {
			cout << "0, QueryUnit QueryStatement" << endl;
			cout << "---------------------------" << endl;
		}
	| QueryStatement {
			cout << "1, QueryStatement" << endl;
			cout << "---------------------------" << endl;
		}
	;

QueryStatement:
	InterMatrix ASSIGN_TOK Multiplier NORMAL_MATMUL_TOK Multiplier {
			cout << "2, InterMatrix ASSIGN_TOK Multiplier NORMAL_MATMUL_TOK Multiplier" << endl;
		}
	| InterMatrix ASSIGN_TOK Multiplier SERMIRING_MATMUL_TOK Multiplier {
			cout << "3, InterMatrix ASSIGN_TOK Multiplier SERMIRING_MATMUL_TOK Multiplier" << endl;
		}
	;

Multiplier:
	STRING_TOK {		// G
			cout << "4, G" << endl;
		}
	| STRING_TOK DOT_TOK STRING_TOK {		// G.T (G.D is not allowed)
			cout << "5, G.T" << endl;
		}
	| STRING_TOK DIM_TOK INT_TOK {		// I^xxxx
			cout << "6, I^xxxx" << endl;
		}
	| STRING_TOK DIM_TOK INT_TOK SCALAR_TOK INT_TOK {		// I^xxxx*xxxx
			cout << "7, I^xxxx*xxxx" << endl;
		}
	| ReformedMatrix {
			cout << "8, ReformedMatrix" << endl;
	}
	| ReformedMatrix SCALAR_TOK INT_TOK {
			cout << "9, ReformedMatrix*xxxx" << endl;
	}
	;

ReformedMatrix:
	InterMatrix {		// m_x_x
			cout << "10, InterMatrix" << endl;
		}
	| InterMatrix DOT_TOK STRING_TOK {		// m_x_x.T or m_x_x.D
			cout << "11, InterMatrix.T/D" << endl;
		}
	| InterMatrix DOT_TOK STRING_TOK DOT_TOK STRING_TOK {		// m_x_x.T.D (other combinations are not allowed)
			cout << "12, InterMatrix.T.D" << endl;		
		}
	;

InterMatrix:
	STRING_TOK UNDERLINE_TOK INT_TOK UNDERLINE_TOK INT_TOK {		// m_x_x
			cout << "13, m_x_x" << endl;
	}
	;

%%

void yyerror(const char *s) {
	cout << "EEK, parse error!  Message: " << s << endl;
	// might as well halt now:
	exit(-1);
}