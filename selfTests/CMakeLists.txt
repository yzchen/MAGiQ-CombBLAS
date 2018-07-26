# Top level directory has the include files

ADD_EXECUTABLE( lubm_scal LubmScal.cpp ../selfInclude/HeaderScal.h)
TARGET_LINK_LIBRARIES( lubm_scal CommGridlib MPITypelib MemoryPoollib Hashlib mmiolib)
