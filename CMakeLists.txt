SET(CMAKE_CXX_COMPILER mpic++)
SET(CMAKE_C_COMPILER mpicc)
INCLUDE(CheckIncludeFileCXX)

cmake_minimum_required(VERSION 3.0)
PROJECT( COMBINATORIAL_BLAS )
SET(CMAKE_INCLUDE_CURRENT_DIR ON)
add_definitions(-DTHREADED)

INCLUDE(TestCXXAcceptsFlag.cmake)
# set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 -std=c++14 -fopenmp -fPIC -O0 -g3 -ggdb")
CHECK_CXX_ACCEPTS_FLAG("-std=c++11 -std=c++14" C14)

OPTION(COMBBLAS_DEBUG
	"Add Macro COMBBLAS_DEBUG"
	OFF)
IF(COMBBLAS_DEBUG)
	MESSAGE("Adding CombBLAS Debug flag...")
	ADD_DEFINITIONS(-DDEBUG)
ENDIF(COMBBLAS_DEBUG)

OPTION(MAGIQ_DEBUG
	"Add Macro MAGIQ_DEBUG"
	OFF)
IF(MAGIQ_DEBUG)
	MESSAGE("Adding MAGiQ Debug flag...")
	ADD_DEFINITIONS(-DMAGIQ_DEBUG)
ENDIF(MAGIQ_DEBUG)

if (NOT C14)
       MESSAGE(FATAL_ERROR "Your compiler does not support C++14")
else (NOT C14)
	if (CMAKE_COMPILER_IS_GNUCXX)
		SET(CMAKE_CXX_FLAGS "-std=c++11 -std=c++14 -fopenmp -DNDEBUG -w -DGRAPH_GENERATOR_SEQ -O2 -DMPICH_IGNORE_CXX_SEEK -D__unused= -DCFG_PARALLEL_USE_GCD=0 -DCFG_PARALLEL_USE_PTHREADS=1")
		SET(CMAKE_C_FLAGS "-DNDEBUG -Drestrict=__restrict__ -w -DGRAPH_GENERATOR_SEQ -O2 -DMPICH_IGNORE_CXX_SEEK -D__unused= -DCFG_PARALLEL_USE_GCD=0 -DCFG_PARALLEL_USE_PTHREADS=1")
	else (CMAKE_COMPILER_IS_GNUCXX)
		CHECK_CXX_ACCEPTS_FLAG(-std=c++0x C0X)
		if (C0X)
			SET(CMAKE_CXX_FLAGS "-std=c++0x -std=c++11 -std=c++14 -fopenmp -DNOTGNU -DNDEBUG -w -DGRAPH_GENERATOR_SEQ -O2 -DMPICH_IGNORE_CXX_SEEK")
			SET(CMAKE_C_FLAGS "-DNOTGNU -DNDEBUG -Drestrict=__restrict__ -w -DGRAPH_GENERATOR_SEQ -O2 -DMPICH_IGNORE_CXX_SEEK")
		else (C0X)
			SET(CMAKE_CXX_FLAGS "-std=c++11 -std=c++14 -DNOTGNU -fopenmp -DNDEBUG -w -DGRAPH_GENERATOR_SEQ -O2 -DMPICH_IGNORE_CXX_SEEK")
			SET(CMAKE_C_FLAGS "-DNOTGNU -DNDEBUG -Drestrict=__restrict__ -w -DGRAPH_GENERATOR_SEQ -O2 -DMPICH_IGNORE_CXX_SEEK")
		endif (C0X)
	endif (CMAKE_COMPILER_IS_GNUCXX)
endif (NOT C14)

ADD_LIBRARY( CommGridlib src/CommGrid.cpp)
ADD_LIBRARY( MPITypelib src/MPIType.cpp)
ADD_LIBRARY( MPIOplib src/MPIOp.cpp)
ADD_LIBRARY( MemoryPoollib src/MemoryPool.cpp)
ADD_LIBRARY( Hashlib src/hash.cpp)
ADD_LIBRARY( mmiolib src/mmio.c)
ADD_LIBRARY( pmergesort external/pmergesort/src/pmergesort.c)

ADD_SUBDIRECTORY( magiq_src )