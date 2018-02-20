#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string>
#include <stdio.h>
#include <map>
#include <vector>

using namespace std;


// Remaps old vector files to new (triples) format
int main(int argc, char *argv[] )
{
	if(argc < 2)
	{
		cout << "Usage: " << argv[0] << " <filename>" << endl;
		return 0; 
	}

	stringstream outs;
	outs << argv[1] << ".remapped";
	ofstream output(outs.str().c_str());
	ifstream input(argv[1]);

	int numrows, total_nnz;
	if (input.is_open())
	{
		int one = 1;
		input >> numrows >> total_nnz;
		output << numrows << "\t" << one << "\t" << total_nnz << endl;
		int cnz = 0;
		int tempind;
		double tempnum;
		while ( (!input.eof()) && cnz < total_nnz)
                {
                    	input >> tempind >> tempnum;
			output << tempind << "\t" << one << "\t" << tempnum << endl;
		}
	}
	input.close();
	output.close();
	
	return 0;
}
