# Top level directory has the include files

ADD_EXECUTABLE( lubm_debug LubmScal.cpp ../selfInclude/HeaderScal.h)
TARGET_LINK_LIBRARIES( lubm_debug CommGridlib MPITypelib MemoryPoollib Hashlib mmiolib /scratch/cheny0l/CombBLAS/build/libpmergesort.a)
