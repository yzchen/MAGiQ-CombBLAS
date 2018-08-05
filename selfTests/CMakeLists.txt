# Top level directory has the include files

ADD_EXECUTABLE( lubm_perm_read_no_tG LubmScal.cpp ../selfInclude/HeaderScal.h)
TARGET_LINK_LIBRARIES( lubm_perm_read_no_tG CommGridlib MPITypelib MemoryPoollib Hashlib mmiolib /scratch/cheny0l/CombBLAS/build/libpmergesort.a)
