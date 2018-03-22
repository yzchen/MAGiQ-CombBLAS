## Clion debug

### Config project in CLion

You just need gcc and openmpi in your system, and system supports cmake, then you can just open a project in CLion.

### Debug settings

You should disable optimizations for compiler, the automatic optimization will confuse a lot when debugging. In addition, open gdb will give you a lot of information needed for debugging.

Can just add following line in `CMakeLists.txt` under base directory.

`set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 -std=c++14 -fopenmp -fPIC -O0 -g3 -ggdb")`

We just set these flags when debugging, when releasing we want all the optimizations.

### Core class definition

#### testing definition

```
typedef SpDCCols<int, NT> DCCols;
typedef SpParMat<int, NT, DCCols> MPI_DCCols;
```

We defined a type named `DCCols` which shows a column type, so here matrix is column-based( **?** ).

Sparse Matrix :

```
A (type : PSpMat<int>::MPI_DCCols)
|
|___CommGrid
|
|___sqSeq(type : SpDCCols<int, int> *)
    |
    |___m, n, nnz
    |
    |___dcsc/dcscarr : values pointers
```

Vector :

```
arr (type : std::vector<int>)
```
