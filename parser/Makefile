CC := g++ -std=c++11 -g
LEX := flex
YACC := bison
CFLAGS := -Wno-duplicate-decl-specifier
YFLAGS :=
LFLAGS :=

all: bin/parser

# compile bison
src/parser.cc include/tok.hh: config/parser.y
	$(YACC) $(YFLAGS) -v --defines=include/tok.hh -o src/parser.cc $<

bin/parser.o: src/parser.cc include/tok.hh
	@mkdir -p bin
	$(CC) $(CFLAGS) -c -o $@ $<

# compile lexer
src/lex.yy.c: config/lexer.l include/tok.hh
	$(LEX) $(LFLAGS) -o $@ $<

bin/lexer.o: src/lex.yy.c
	@mkdir -p bin
	$(CC) $(CFLAGS) -c -o $@ $<

# main
bin/main.o: src/main.cc include/tok.hh
	@mkdir -p bin
	$(CC) $(CFLAGS) -c -o $@ $<

# final
bin/parser: bin/lexer.o bin/parser.o bin/main.o
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: clean
clean:
	rm -f bin/*.o src/lex.yy.c src/parser.cc include/tok.hh
	rm -f *.dot
	rm -f *~ .*~