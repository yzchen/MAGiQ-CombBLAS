# Top level directory has the include files

ADD_EXECUTABLE( lubm10240 Lubm10240.cpp ../include/Header10240.h)
ADD_EXECUTABLE( encoded_lubm10240 Lubm10240_Encoded.cpp ../include/Header10240.h)
ADD_EXECUTABLE( lubm100k Lubm100k.cpp ../include/Header100k.h)
ADD_EXECUTABLE( encoded_lubm1T Lubm1T_Encoded.cpp ../include/Header1T.h)

TARGET_LINK_LIBRARIES( lubm10240 CommGridlib MPITypelib MemoryPoollib Hashlib mmiolib)
TARGET_LINK_LIBRARIES( encoded_lubm10240 CommGridlib MPITypelib MemoryPoollib Hashlib mmiolib)
TARGET_LINK_LIBRARIES( lubm100k CommGridlib MPITypelib MemoryPoollib Hashlib mmiolib)
TARGET_LINK_LIBRARIES( encoded_lubm1T CommGridlib MPITypelib MemoryPoollib Hashlib mmiolib)
