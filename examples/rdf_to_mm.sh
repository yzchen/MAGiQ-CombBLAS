#! /bin/bash

# Convert an rdf file in our format to Matrix Market format used by CombBLAS
# $1: encoded.nt file with:
# EXAMPLE
# num_nonzeroes
# num_nodes
# <s> <p> <o> .
# .
# .
# $2: output file

if [ $# -eq 4 ]; then
    echo "Number of nodes and edges supplied."
    echo "Input: $1"
    echo "Output: $2"
    echo "Nodes: $3"
    echo "Edges: $4"
    echo "%%MatrixMarket matrix coordinate real general" > $2
    echo "$3	$3	$4" >> $2
    sed "s/<\(.*\)> <\(.*\)> <\(.*\)> ./\1\t\3\t\2/g" $1 | awk '{print $1+1"\t"$2+1"\t"$3}' >> $2
elif [ $# -eq 2 ]; then
    echo "Number of nodes and edges to be read from input file."
    echo "Input: $1"
    echo "Output: $2"
    echo "%%MatrixMarket matrix coordinate real general" > $2
    head -2 $1 | tr '\n' ' ' | sed "s/\([0-9]*\) \([\0-9]*\) /\2\t\2\t\1\n/" >> $2
    tail -n+3 $1 | sed "s/<\(.*\)> <\(.*\)> <\(.*\)> ./\1\t\3\t\2/g" | awk '{print $1+1"\t"$2+1"\t"$3}' >> $2
else
    echo "Usage: "
    echo "./script inpath outpath OR ./script inpath outpath #nodes #edges"
fi

# echo "%%MatrixMarket matrix coordinate real general" > $2
# head -2 $1 | tr '\n' ' ' | sed "s/\([0-9]*\) \([\0-9]*\) /\2\t\2\t\1\n/" >> $2
# tail -n+3 $1 | sed "s/<\(.*\)> <\(.*\)> <\(.*\)> ./\1\t\3\t\2/g" | awk '{print $1+1"\t"$2+1"\t"$3}' >> $2

