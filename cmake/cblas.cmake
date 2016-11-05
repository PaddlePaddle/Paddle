# Find the CBlas and lapack libraries
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

find_path(MKL_INCLUDE_DIR mkl.h PATHS
  ${MKL_ROOT}/include)
find_path(MKL_INCLUDE_DIR mkl_lapacke.h PATHS
  ${MKL_ROOT}/include)
find_library(MKL_CORE_LIB NAMES mkl_core PATHS
  ${MKL_ROOT}/lib
  ${MKL_ROOT}/lib/intel64)
find_library(MKL_SEQUENTIAL_LIB NAMES mkl_sequential PATHS
  ${MKL_ROOT}/lib
  ${MKL_ROOT}/lib/intel64)
find_library(MKL_INTEL_LP64 NAMES mkl_intel_lp64 PATHS
  ${MKL_ROOT}/lib
  ${MKL_ROOT}/lib/intel64)


if(MKL_INCLUDE_DIR AND MKL_CORE_LIB AND MKL_SEQUENTIAL_LIB AND MKL_INTEL_LP64)
  set(CBLAS_PROVIDER MKL)
  set(CBLAS_INC_DIR ${MKL_INCLUDE_DIR})
  set(CBLAS_LIBS ${MKL_INTEL_LP64}
          ${MKL_SEQUENTIAL_LIB}
          ${MKL_CORE_LIB})
  add_definitions(-DPADDLE_USE_MKL)
  message(STATUS "Found MKL (include: ${CBLAS_INC_DIR}, library: ${CBLAS_LIBS})")
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
find_path(ATLAS_CLAPACK_INC_DIR NAMES clapack.h
  PATHS ${ATLAS_INCLUDE_SEARCH_PATHS})
find_library(ATLAS_CBLAS_LIB NAMES cblas libcblas.so.3 
  PATHS ${ATLAS_LIB_SEARCH_PATHS})
find_library(ATLAS_LIB NAMES lapack_atlas liblapack_atlas.so.3
  PATHS ${ATLAS_LIB_SEARCH_PATHS})

if(ATLAS_INC_DIR AND ATLAS_CBLAS_LIB AND ATLAS_LIB)
  set(CBLAS_PROVIDER ATLAS)
  set(CBLAS_INC_DIR ${ATLAS_INC_DIR} ${ATLAS_CLAPACK_INC_DIR})
  set(CBLAS_LIBS ${ATLAS_LIB} ${ATLAS_CBLAS_LIB})
  add_definitions(-DPADDLE_USE_ATLAS)  
  message(STATUS "Found Atlas (include: ${CBLAS_INC_DIR}, library: ${CBLAS_LIBS})")
  return()
endif()

## Then find openblas.
set(OPENBLAS_ROOT $ENV{OPENBLAS_ROOT} CACHE PATH "Folder contains Openblas")
set(OPENBLAS_INCLUDE_SEARCH_PATHS
        ${OPENBLAS_ROOT}/include
        /usr/include
        /usr/include/openblas
        /usr/local/opt/openblas/include)
set(OPENBLAS_LIB_SEARCH_PATHS
        ${OPENBLAS_ROOT}/lib
        /usr/lib
        /usr/lib/blas/openblas
        /usr/lib/openblas
        /usr/local/opt/openblas/lib)

find_path(OPENBLAS_INC_DIR NAMES cblas.h
  PATHS ${OPENBLAS_INCLUDE_SEARCH_PATHS})
find_path(OPENBLAS_LAPACKE_INC_DIR NAMES lapacke.h
  PATHS ${OPENBLAS_INCLUDE_SEARCH_PATHS})
find_library(OPENBLAS_LIB NAMES openblas
  PATHS ${OPENBLAS_LIB_SEARCH_PATHS})

if(OPENBLAS_INC_DIR AND OPENBLAS_LIB)
  set(CBLAS_PROVIDER OPENBLAS)
  set(CBLAS_INC_DIR ${OPENBLAS_INC_DIR})
  set(CBLAS_LIBS ${OPENBLAS_LIB})
  message(STATUS "Found OpenBlas (include: ${CBLAS_INC_DIR}, library: ${CBLAS_LIBS})")
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
