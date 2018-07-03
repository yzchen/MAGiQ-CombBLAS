## Lubm10240-L5-ResGen (local 16 processes)

```
###############################################################
Load Matrix
###############################################################
---------------------------------------------------------------
starting reading lubm10240 data......
Matrix is Float
Total number of nonzeros expected across all processors is 1366712443
File is 25140824453 bytes
As a whole: 336506397 rows and 336506397 columns and 1366485519 nonzeros
	read file takes : 120.937 s
	original imbalance of G : 1.99284
	permutation takes : 219.241 s
	imbalance of permuted G : 1.34314
	transpose G takes : 10.4195 s
graph load (Total) : 352.949 s
---------------------------------------------------------------

###############################################################
Query 5
###############################################################
---------------------------------------------------------------
step 1 : m_(2,0) = G x {1@(11,11)}*11
	dim-apply takes: 0.531108 s
	prune takes: 0.350906 s
step 1 (Total) : 0.882108 s
---------------------------------------------------------------
step 2 : G.T() x m_(2,0).D()*6
	diag-reduce takes : 0.885064 s
	mmul-scalar takes : 0.032599 s
	dim-apply takes: 0.667519 s
	prune takes: 0.586764 s
step 2 (Total) : 2.17226 s
---------------------------------------------------------------
step 3 : m_(1,2) = {1@(357,357)} x m_(1,2)
	dim-apply takes: 0.156441 s
	prune takes: 2.14577e-06 s
step 3 (Total) : 0.156661 s
---------------------------------------------------------------
step 4 : m_(2,0) = m_(1,2).T().D() x m_(2,0)
	diag-reduce takes : 1.15936 s
	mmul-scalar takes : 0 s
	dim-apply takes: 0.409237 s
	prune takes: 2.14577e-06 s
step 4 (Total) : 1.56872 s
---------------------------------------------------------------
10 [ 10, 1 ]
	enum takes 1.23981 s
	imbalance : 6.4
---------------------------------------------------------------
query5 mmul_scalar time : 0.032599 s
query5 prune time : 0.937674 s
query5 diag_reduce time : 2.04442 s
query5 dim_apply time : 1.76431 s
query5 time (Total) : 4.77996 s
---------------------------------------------------------------
begin result generation ......
0 1 1 1 1
0 size of res : 1
final size : 4
final size : 8
final size : 10
4 3 3 3 3
4 size of res : 3
8 4 4 4 4
8 size of res : 4
12 2 2 2 2
12 size of res : 2
```