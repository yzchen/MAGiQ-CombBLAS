## Report 2 : Basic functions implementation in Comb_BLAS

### Task description

Please do the following task before moving on to the tasks below:

0. Write a program that takes a matrix in matrix market format as an input and adds the elements in each row and writes the resulting vector to disk (a reduction with the ADD operator along the rows).

I have produced some example matrices in /home/cheny0l/work/db245/fuad/data/ on your machine to test this program with.
I will call your program as follows:

`mpirun -np 16 row_reduce_yan input_matrix output_path`

The following are to be done using CombBLAS functions

1. Write a function set_element(M, i, j, v) that takes a matrix M and sets M(i, j) to v

2. Write a function mmul_scalar(M, s) that multiplies matrix M with scalar s

3. write a function diagonalize(M) that does a reduction along each row with the OR operator and places the resulting vector on the diagonal of M

4. write a function transpose(M) that transposes M

### Environment change log

I put all my tests in `selfTests` folder, so the executable files will be `build/selfTests`.

For example, if you want to run the `spmult`, you can :

```
cd build

./selfTests/spmult mat1.file mat2.file
```

In addition, `TESTDATA` is in the project home directory, previous it's in `build`, and I remove the tracking for `build` directory in git.

External test data is in `../fuad/data`

### Generate matrix market by yourself

I wrote a script to generate a matrix file, it locates in `TESTDATA` folder, and you should first go to that folder

`cd TESTDATA`

Then run the command :

`julia create_mm.jl m n nnz`

you should specify `m`, `n` and `nnz`, and you will get a txt file named as `gen_m_n_nnz.txt`.

We can generate any dimensions for test, it is convenient.

### Task 0 : reduce add

Basically call the internal `Reduce` function is fine, some problems and issues should be considered :

* In matrix market format, index should start from 1 not 0

    ---> Why this matters? Because there is a `delete []` in the `Reduce`, if real data starts from 0, there will be an index is outside the array bound, which is terrible.

* **?** Duplicate edges between same vertex pair are not supported

    ---> I created a simple matrix market format file that has duplicate edge (`TESTDATA/duplicate_edges.txt`), file content :

    ![dup_edges](./imgs/report2/dup_edges.jpeg)

    Duplicate position is `8 3`

    Sparse matrix construction from reading file is fine with no errors. Apply `reduceadd` to that matrix, result :

    ![reduce_add_dup_edges](./imgs/report2/reduce_add_dup_edges.jpeg)

    Two things should be mentioned in this result :

    1. Matrix construction will treat duplicate edges as different edges, because they has `10` non-zeros (probably because the sparse matrix storage is different from the normal matrix array storage)

    2. ReduceAdd can work correctly in this case, row `8` has correct summation.

    In addition, I tried to construct sparse matrix from scratch (pre-define dimension and non-zero positions and values). The code file is `selfTests/Construct_SpMat.cpp`. The result is :

    ![construct_mat](./imgs/report2/construct_mat.png)

    (Basically the construction from code is correct, you set `sumDuplicate` and get `sum` of duplicate values)

### ReduceAdd on lubm data

#### Without setting mpi #procs

When I ran test on `lubm2560` and `lubm10240` dataset, error occurred,

![lubm_error](./imgs/report2/lubm_error.png)

But I got result without errors with `lubm320` data,

![lubm320_done](./imgs/report2/lubm320_done.png)

dimension (#rows = #columns) of three data :

|lubm320|lubm2560|lubm10240|
|:-----:|:------:|:-------:|
|10487737|84151528|336506397|

I doubt if is the size problem, of course a debug about `Reduce` function is needed.

#### Setting mpi #procs = 16

![lubm320_done_np16](./imgs/report2/lubm320_done_np16.png)

![lubm_error_np16](./imgs/report2/lubm_error_np16.png)

The problem here is about `double free or corruption`, without setting mpi, the error is `address not mapped`.

All errors are because the start index is 0 not 1, I generated some matrixes myself and passed the test for both cases.

`scale_test.sh` in `TESTDATA` can automatically generate different nnz matrix for testing, I fixed dimension with `80000000 * 80000000`, change nnz from `150000000` to `300000000`, everything is okay. Then I generated an example with the same size with largest lubm data and same nnz. Following picture shows the result.

![generated_data_done](./imgs/report2/generated_data_done.png)

### Task 1-4 : utility functions

Result together:

![task1-4](./imgs/report2/task1-4.png)

`SaveGethered` function can store sparse matrix to a file with tripple format.

#### Set element

There are two methods to achieve this,

* Construct a 1*1 matrix based on the parameter `v`, and then use `SpAsgn` function to assign a block to matrix. This algorithm has some problem when I set the length of index vectors to `1`, but it can work when I set them more than `1`. I don't know why.

* Based on the index `i, j` construct two vectors, both vectors only have `1` value which is `i/j`; construct a 1*1 matrix based on the parameter `v`, but its dimension is same as the matrix. Then call `Prune` to clear given position, and then add two matrices, this is the method I adopted.

One thing should be concerned, before construct 1×1 matrix, before pruning, before adding, values of index should decrease `1`.

`ri.Apply(bind2nd(minus<int>(), 1));`

![setElement](./imgs/report2/setElement.png)

#### Multiply a scalar

Use the `Apply` function,

`Apply(bind2nd(multiplies<ElementType>(), s));`

Before multiplication:

![mmul_scalar_origin](./imgs/report2/mmul_scalar_origin.png)

After multiplication:

![mmul_scalar](./imgs/report2/mmul_scalar.png)

### Diagonalize

First use `std::logical_or` to reduce all rows of input matrix, and then construct a sparse matrix with same dimension with input matrix, call `set_element` to set the diagonal of the matrix.

![diagonalize](./imgs/report2/diagonalize.png)

Try to use `SpAsgn` to assgin a vector to diagonal of matrix.

Problems about `SpAsgn`:

![spasgn](./imgs/report2/spasgn.png)

Input index vector `riv` and `civ` are same, so it should add two values at `(2, 2)` and `(9, 9)` with value `100000` and `999999`, but result is not what we want, `SpAsgn` use the all combinations of `riv` and `civ` to assign values. **?** I don't know why they design this function like this.

The reason is this action needs a `Prune` first, `Prune` is implemented by `SAT`, `S(m,m,ri,ri,1)`, `T(n,n,ci,ci,1)`. If `ri = (2, 9) = ci`, then this multiplication just does selection, so `4` values will be selected, their positions are `(2,2), (2,9), (9,2), (9,9)`.
Then `A * logical_not(SAT)` will Prune values in these `4` positions.

I think this `SpAsgn` is useful in undirected graph operations.

A simple way to assgin vector to matrix diagonal is just construction... `D(dim, dim, *rvec, *qvec, diag)`, `*rvec = *qvec` both contains values from `1` to `dimOf(A)`.

### Transpose

Just call internal `Transpose` function

![transpose](./imgs/report2/transpose.png)
