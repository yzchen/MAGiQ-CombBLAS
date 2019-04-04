# Top level directory has the include files

ADD_EXECUTABLE( magiqScal magiqScal.cpp ../magiq_include/magiqScal.h)
TARGET_LINK_LIBRARIES( magiqScal CommGridlib MPITypelib MemoryPoollib Hashlib mmiolib pmergesort)

ADD_EXECUTABLE( magiqParse magiqParse.cc ../magiq_include/magiqScal.h)
TARGET_LINK_LIBRARIES( magiqParse CommGridlib MPITypelib MemoryPoollib Hashlib mmiolib pmergesort)