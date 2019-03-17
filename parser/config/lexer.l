%{
	#include <cstdio>
	#include <cstdlib>
	#include <string>

	#include "../include/tok.hh"
%}

%option noyywrap

D		[0-9]
L		[a-zA-Z]

%%

"#".*				// ignore comment
"-".*				// ignore comment

{D}+				{ yylval.ival = atoi(yytext); return INT_TOK; }		// numbers

{L}({L}|{D})*		{ yylval.sval = new std::string(yytext); return STRING_TOK; }		// identifiers

"_"					return UNDERLINE_TOK;		// indicate underline 

"="					return ASSIGN_TOK;		// assignment op

"^"					return DIM_TOK;		// indicate dimension of identity matrix

"."					return DOT_TOK;		// dot op, for transpose and diagonalize

"*"					return SCALAR_TOK;		// matrix times integer

"×"					return NORMAL_MATMUL_TOK;		// normal matrix multiplication, × is not x, they are different

"⊗"					return SERMIRING_MATMUL_TOK;		// sermiring matrix multiplication

[ \t\n]*           	;

.					return ERROR_TOK;
%%