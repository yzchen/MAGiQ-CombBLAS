# CMake generated Testfile for 
# Source directory: /home/cheny0l/work/db245/CombBLAS_beta_16_1
# Build directory: /home/cheny0l/work/db245/CombBLAS_beta_16_1/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(Multiplication_Test "mpirun" "-np" "4" "./ReleaseTests/MultTest" "TESTDATA/rmat_scale16_A.mtx" "TESTDATA/rmat_scale16_B.mtx" "TESTDATA/rmat_scale16_productAB.mtx" "TESTDATA/x_65536_halfdense.txt" "TESTDATA/y_65536_halfdense.txt")
add_test(Reduction_Test "mpirun" "-np" "4" "./ReleaseTests/ReduceTest" "TESTDATA/sprand10000" "TESTDATA/sprand10000_sumcols" "TESTDATA/sprand10000_sumrows")
add_test(Iterator_Test "mpirun" "-np" "4" "./ReleaseTests/IteratorTest" "TESTDATA" "sprand10000")
add_test(Transpose_Test "mpirun" "-np" "4" "./ReleaseTests/TransposeTest" "TESTDATA" "betwinput_scale16" "betwinput_transposed_scale16")
add_test(Indexing_Test "mpirun" "-np" "4" "./ReleaseTests/IndexingTest" "TESTDATA" "B_100x100.txt" "B_10x30_Indexed.txt" "rand10outta100.txt" "rand30outta100.txt")
add_test(SpAsgn_Test "mpirun" "-np" "4" "./ReleaseTests/SpAsgnTest" "TESTDATA" "A_100x100.txt" "A_with20x30hole.txt" "dense_20x30matrix.txt" "A_wdenseblocks.txt" "20outta100.txt" "30outta100.txt")
add_test(GalerkinNew_Test "mpirun" "-np" "4" "./ReleaseTests/GalerkinNew" "TESTDATA/grid3d_k5.txt" "TESTDATA/offdiag_grid3d_k5.txt" "TESTDATA/diag_grid3d_k5.txt" "TESTDATA/restrict_T_grid3d_k5.txt")
add_test(ParIO_Test "mpirun" "-np" "4" "./ReleaseTests/ParIOTest" "TESTDATA/sevenvertex.mtx" "TESTDATA/sevenvertexgraph.txt")
add_test(FindSparse_Test "mpirun" "-np" "4" "./ReleaseTests/FindSparse" "TESTDATA" "findmatrix.txt")
add_test(BetwCent_Test "mpirun" "-np" "4" "./Applications/betwcent" "TESTDATA/SCALE16BTW-TRANSBOOL/" "10" "96")
add_test(TopDownBFS_Test "mpirun" "-np" "4" "./Applications/tdbfs" "Force" "17" "FastGen")
add_test(DirOptBFS_Test "mpirun" "-np" "4" "./Applications/dobfs" "17")
add_test(FBFS_Test "mpirun" "-np" "4" "./Applications/fbfs" "Gen" "16")
add_test(FMIS_Test "mpirun" "-np" "4" "./Applications/fmis" "17")
add_test(RCM_Test "mpirun" "-np" "4" "./Ordering/rcm" "er" "18")
add_test(SpMSpVBench_test "mpirun" "-np" "4" "./Applications/SpMSpV-IPDPS2017/SpMSpVBench" "-rmat" "18")
subdirs(ReleaseTests)
subdirs(Applications)
subdirs(usort)
subdirs(graph500-1.2/generator)
subdirs(Ordering)
subdirs(Applications/SpMSpV-IPDPS2017)
