%{
	#include <cstdio>
	#include <cstdlib>
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

%token INT_TOK UNDERLINE_TOK ASSIGN_TOK DIM_TOK DOT_TOK SCALAR_TOK NORMAL_MATMUL_TOK SERMIRING_MATMUL_TOK ERROR_TOK
%token STRING_TOK

%type <ival> INT_TOK
%type <sval> STRING_TOK Multiplier InterMatrix ReformedMatrix QueryStatement

%%

QueryUnit:
		{
			cout << "0, " << endl;
			cout << "---------------------------" << endl;			
		}
	| QueryUnit QueryStatement {
			cout << "1, QueryUnit QueryStatement" << endl;
			cout << "---------------------------" << endl;
		}
	;

QueryStatement:
	InterMatrix ASSIGN_TOK Multiplier NORMAL_MATMUL_TOK Multiplier {
			// always allocate new space for literals, 
			// otherwise compiler would use memroy pool, it would course memory overwrite problem
			$$ = new string((*$1) + " = " + (*$3) + " × " + (*$5));
			cout << "2, *> " << *$$ << endl;
		}
	| InterMatrix ASSIGN_TOK Multiplier SERMIRING_MATMUL_TOK Multiplier {
			$$ = new string((*$1) + " = " + (*$3) + " ⊗ " + (*$5));
			cout << "3, %> " << *$$ << endl;
		}
	;

Multiplier:
	STRING_TOK {		// G
			$$ = new string(*$1);
			if ((*$$).compare("G") != 0) {
				cout << "parser error, only G is allowed for rdf data sparse matrix" << endl;
				exit(-1);
			}
			cout << "4, -> " << *$$ << endl;
		}
	| STRING_TOK DOT_TOK STRING_TOK {		// G.T (G.D is not allowed)
			$$ = new string((*$1) + string(".") + (*$3));
			if ((*$$).compare("G.T") != 0) {
				cout << "parser error, only G.T is allowed here" << endl;
				exit(-1);
			}
			cout << "5, => " << *$$ << endl;
		}
	| STRING_TOK DIM_TOK INT_TOK {		// I^xxxx
			if ((*$1).compare("I") != 0) {
				cout << "parser error, only I is allowed here" << endl;
				exit(-1);
			}

			// TODO: construct a identity matrix I with dimension $3(int)
			// TODO: construct a identity matrix I with dimension $3(int)
			$$ = new string((*$1) + "^" + to_string($3));
			cout << "6, !> " << *$$ << endl;
		}
	| STRING_TOK DIM_TOK INT_TOK SCALAR_TOK INT_TOK {		// I^xxxx*xxxx
			if ((*$1).compare("I") != 0) {
				cout << "parser error, only I is allowed here" << endl;
				exit(-1);
			}
			// TODO: construct a diagonal matrix I with value $5(int) with dimension $3(int)
			// TODO: construct a diagonal matrix I with value $5(int) with dimension $3(int)
			$$ = new string((*$1) + "^" + to_string($3) + "*" + to_string($5));
			cout << "7, +> " << *$$ << endl;
		}
	| ReformedMatrix {
			$$ = new string(*$1);
			cout << "8, &> " << *$$ << endl;
	}
	| ReformedMatrix SCALAR_TOK INT_TOK {
			$$ = new string((*$1) + "*" + to_string($3));
			cout << "9, ^> " << *$$ << endl;
	}
	;

ReformedMatrix:
	InterMatrix {		// m_x_x
			$$ = new string(*$1);
			cout << "10, '> " << *$$ << endl;
		}
	| InterMatrix DOT_TOK STRING_TOK {		// m_x_x.T or m_x_x.D
			// TODO: apply reformation to intermediate matrix
			// TODO: apply reformation to intermediate matrix
			$$ = new string((*$1) + "." + (*$3));
			cout << "11, _> " << *$$ << endl;
		}
	| InterMatrix DOT_TOK STRING_TOK DOT_TOK STRING_TOK {		// m_x_x.T.D (other combinations are not allowed)
			if ((*$3).compare("T") == 0 && (*$5).compare("D") == 0) {
				// TODO: apply reformation to intermediate matrix
				// TODO: apply reformation to intermediate matrix
				$$ = new string((*$1) + "." + (*$3) + "." + (*$5));
				cout << "12, :> " << *$$ << endl;
			} else {
				cout << "parser error, only .T.D combination is allowed" << endl;
				exit(-1);
			}
		}
	;

InterMatrix:
	STRING_TOK UNDERLINE_TOK INT_TOK UNDERLINE_TOK INT_TOK {		// m_x_x
			if ((*$1).compare("m") != 0) {
				cout << "parser error, only m is allowed here" << endl;
				exit(-1);
			}
			$$ = new string((*$1) + "_" + to_string($3) + "_" + to_string($5));

			// TODO: construct intermediate matrix
			// TODO: construct intermediate matrix
			cout << "13, ~> " << *$$ << endl;
	}
	;

%%

void yyerror(const char *s) {
	cout << "EEK, parse error!  Message: " << s << endl;
	// might as well halt now:
	exit(-1);
}