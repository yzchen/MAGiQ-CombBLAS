# Top level directory has the include files

ADD_EXECUTABLE( spmult SpMult.cpp )
ADD_EXECUTABLE( reduceadd ReduceAdd.cpp )
ADD_EXECUTABLE( utils utils.cpp )
ADD_EXECUTABLE( constructmat Construct_SpMat.cpp )

TARGET_LINK_LIBRARIES( spmult CommGridlib MPITypelib MemoryPoollib Hashlib mmiolib)
TARGET_LINK_LIBRARIES( reduceadd CommGridlib MPITypelib MemoryPoollib Hashlib mmiolib)
TARGET_LINK_LIBRARIES( utils CommGridlib MPITypelib MemoryPoollib Hashlib mmiolib)
TARGET_LINK_LIBRARIES( constructmat CommGridlib MPITypelib MemoryPoollib Hashlib mmiolib)
