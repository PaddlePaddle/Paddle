# Find the CBlas libraries
#
# It will search MKL, atlas, OpenBlas, reference-cblas in order.
#
# If any cblas implementation found, the following variable will be set.
#    CBLAS_PROVIDER  # one of MKL, ATLAS, OPENBLAS, REFERENCE
#    CBLAS_INC_DIR   # the include directory for cblas.
#    CBLAS_LIBS      # a list of libraries should be linked by paddle. 
#                    # Each library should be full path to object file.
#
# User should set one of MKL_ROOT, ATLAS_ROOT, OPENBLAS_ROOT, REFERENCE_CBLAS_ROOT
# during cmake. If none of them set, it will try to find cblas implementation in
# system paths.
#


## Find MKL First.
set(MKL_ROOT $ENV{MKL_ROOT} CACHE PATH "Folder contains MKL")

find_path(MKL_INCLUDE_DIR mkl.h PATHS ${MKL_ROOT}/include)
find_library(MKL_CORE_LIB NAMES mkl_core PATHS ${MKL_ROOT}/lib)
find_library(MKL_SEQUENTIAL_LIB NAMES mkl_sequential PATHS ${MKL_ROOT}/lib)
find_library(MKL_INTEL_LP64 NAMES mkl_intel_lp64 PATHS ${MKL_ROOT}/lib)


if(MKL_INCLUDE_DIR AND MKL_CORE_LIB AND MKL_SEQUENTIAL_LIB AND MKL_INTEL_LP64)
  set(CBLAS_PROVIDER MKL)
  set(CBLAS_INC_DIR ${MKL_INCLUDE_DIR})
  set(CBLAS_LIBS ${MKL_INTEL_LP64}
          ${MKL_SEQUENTIAL_LIB}
          ${MKL_CORE_LIB})
  add_definitions(-DPADDLE_USE_MKL)
  return() # return file.
endif()

## Then find atlas.
set(ATLAS_ROOT $ENV{ATLAS_ROOT} CACHE PATH "Folder contains Atlas")
set(ATLAS_INCLUDE_SEARCH_PATHS
        ${ATLAS_ROOT}/include
        /usr/include
        /usr/include/atlas)
set(ATLAS_LIB_SEARCH_PATHS
        ${ATLAS_ROOT}/lib
        /usr/lib
        /usr/lib/blas/atlas
        /usr/lib/atlas
        /usr/lib/atlas-base   # special for ubuntu 14.04.
    )
find_path(ATLAS_INC_DIR NAMES cblas.h 
  PATHS ${ATLAS_INCLUDE_SEARCH_PATHS})
find_library(ATLAS_CBLAS_LIB NAMES cblas libcblas.so.3 
  PATHS ${ATLAS_LIB_SEARCH_PATHS})
find_library(ATLAS_LIB NAMES atlas libatlas.so.3
  PATHS ${ATLAS_LIB_SEARCH_PATHS})

if(ATLAS_INC_DIR AND ATLAS_CBLAS_LIB AND ATLAS_LIB)
  set(CBLAS_PROVIDER ATLAS)
  set(CBLAS_INC_DIR ${ATLAS_INC_DIR})
  set(CBLAS_LIBS ${ATLAS_LIB} ${ATLAS_CBLAS_LIB})
  return()
endif()

## Then find openblas.
SET(Open_BLAS_INCLUDE_SEARCH_PATHS
  /usr/include
  /usr/include/openblas
  /usr/include/openblas-base
  /usr/local/opt/openblas/include
  /usr/local/include
  /usr/local/include/openblas
  /usr/local/include/openblas-base
  /opt/OpenBLAS/include
  $ENV{OpenBLAS_HOME}
  $ENV{OpenBLAS_HOME}/include
)

SET(Open_BLAS_LIB_SEARCH_PATHS
        /lib/
        /lib/openblas-base
        /lib64/
        /usr/lib
        /usr/lib/openblas-base
		/usr/local/opt/openblas/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /opt/OpenBLAS/lib
        $ENV{OpenBLAS}cd
        $ENV{OpenBLAS}/lib
        $ENV{OpenBLAS_HOME}
        $ENV{OpenBLAS_HOME}/lib
 )

FIND_PATH(OpenBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${Open_BLAS_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(OpenBLAS_LIB NAMES openblas PATHS ${Open_BLAS_LIB_SEARCH_PATHS})

SET(OpenBLAS_FOUND ON)

#    Check include files
IF(NOT OpenBLAS_INCLUDE_DIR)
    SET(OpenBLAS_FOUND OFF)
    MESSAGE(STATUS "Could not find OpenBLAS include. Turning OpenBLAS_FOUND off")
ENDIF()

#    Check libraries
IF(NOT OpenBLAS_LIB)
    SET(OpenBLAS_FOUND OFF)
    MESSAGE(STATUS "Could not find OpenBLAS lib. Turning OpenBLAS_FOUND off")
ENDIF()

IF (OpenBLAS_FOUND)
  IF (NOT OpenBLAS_FIND_QUIETLY)
    MESSAGE(STATUS "Found OpenBLAS libraries: ${OpenBLAS_LIB}")
    MESSAGE(STATUS "Found OpenBLAS include: ${OpenBLAS_INCLUDE_DIR}")
  ENDIF (NOT OpenBLAS_FIND_QUIETLY)
ELSE (OpenBLAS_FOUND)
  IF (OpenBLAS_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find OpenBLAS")
  ENDIF (OpenBLAS_FIND_REQUIRED)
ENDIF (OpenBLAS_FOUND)

MARK_AS_ADVANCED(
    OpenBLAS_INCLUDE_DIR
    OpenBLAS_LIB
    OpenBLAS
)

if(OpenBLAS_INCLUDE_DIR AND OpenBLAS_LIB)
  set(CBLAS_PROVIDER OPENBLAS)
  set(CBLAS_INC_DIR ${OpenBLAS_INCLUDE_DIR})
  set(CBLAS_LIBS ${OpenBLAS_LIB})
  return()
endif()


## Then find the reference-cblas.  www.netlib.org/blas/


set(REFERENCE_CBLAS_ROOT $ENV{REFERENCE_CBLAS_ROOT} CACHE PATH 
  "Folder contains reference-cblas")
set(REFERENCE_CBLAS_INCLUDE_SEARCH_PATHS
  ${REFERENCE_CBLAS_ROOT}/include
  /usr/include
  /usr/include/cblas
)

set(REFERENCE_CBLAS_LIB_SEARCH_PATHS
  ${REFERENCE_CBLAS_ROOT}/lib
  /usr/lib
  /usr/lib/blas/reference/
  /usr/lib/reference/
)

find_path(REFERENCE_CBLAS_INCLUDE_DIR NAMES cblas.h PATHS
        ${REFERENCE_CBLAS_INCLUDE_SEARCH_PATHS})
find_library(REFERENCE_CBLAS_LIBRARY NAMES cblas PATHS
        ${REFERENCE_CBLAS_LIB_SEARCH_PATHS})

if (REFERENCE_CBLAS_INCLUDE_DIR AND REFERENCE_CBLAS_LIBRARY)
  set(CBLAS_PROVIDER REFERENCE)
  set(CBLAS_INC_DIR ${REFERENCE_CBLAS_INCLUDE_DIR})
  set(CBLAS_LIBS ${REFERENCE_CBLAS_LIBRARY})
  return()
endif()

message(FATAL_ERROR "CBlas must be set. Paddle support MKL, ATLAS, OpenBlas, reference-cblas."
  " Try set MKL_ROOT, ATLAS_ROOT, OPENBLAS_ROOT or REFERENCE_CBLAS_ROOT.")
