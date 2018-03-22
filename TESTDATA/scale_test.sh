#!/bin/bash

echo "testing for 80000000 * 80000000 with different nnz matrixes"

listVar=(150000000 200000000 250000000 300000000)
for nnz in ${listVar[@]}
do
    echo "    test for nnz : ${nnz}"
    julia create_mm.jl 80000000 80000000 ${nnz}
    ../build/selfTests/reduceadd gen_80000000_80000000_${nnz}.txt output_${nnz}.del
done

echo "all tests done"
