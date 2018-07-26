# Top level directory has the include files

ADD_EXECUTABLE( lubm_scal_freemem LubmScal.cpp ../selfInclude/HeaderScal.h)
TARGET_LINK_LIBRARIES( lubm_scal_freemem CommGridlib MPITypelib MemoryPoollib Hashlib mmiolib)
