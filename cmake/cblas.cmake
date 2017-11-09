# Find the CBlas and lapack libraries
#
# It will search MKLML, atlas, OpenBlas, reference-cblas in order.
#
# If any cblas implementation found, the following variable will be set.
#    CBLAS_PROVIDER  # one of MKLML, ATLAS, OPENBLAS, REFERENCE
#    CBLAS_INC_DIR   # the include directory for cblas.
#    CBLAS_LIBS      # a list of libraries should be linked by paddle.
#                    # Each library should be full path to object file.

set(CBLAS_FOUND OFF)

## Find MKLML First.
if(WITH_MKLML AND MKLML_INC_DIR AND MKLML_LIB)
  set(CBLAS_FOUND ON)
  set(CBLAS_PROVIDER MKLML)
  set(CBLAS_INC_DIR ${MKLML_INC_DIR})
  set(CBLAS_LIBRARIES ${MKLML_LIB})

  add_definitions(-DPADDLE_USE_MKLML)
  add_definitions(-DLAPACK_FOUND)

  message(STATUS "Found cblas and lapack in MKLML "
    "(include: ${CBLAS_INC_DIR}, library: ${CBLAS_LIBRARIES})")
  return()
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
find_library(ATLAS_CLAPACK_LIB NAMES lapack_atlas liblapack_atlas.so.3
  PATHS ${ATLAS_LIB_SEARCH_PATHS})

if(ATLAS_CLAPACK_INC_DIR AND ATLAS_INC_DIR AND ATLAS_CBLAS_LIB AND ATLAS_CLAPACK_LIB)
  set(CBLAS_FOUND ON)
  set(CBLAS_PROVIDER ATLAS)
  set(CBLAS_INC_DIR ${ATLAS_INC_DIR} ${ATLAS_CLAPACK_INC_DIR})
  set(CBLAS_LIBRARIES ${ATLAS_CLAPACK_LIB} ${ATLAS_CBLAS_LIB})

  add_definitions(-DPADDLE_USE_ATLAS)
  add_definitions(-DLAPACK_FOUND)

  message(STATUS "Found ATLAS (include: ${ATLAS_INC_DIR}, library: ${CBLAS_LIBRARIES})")
  message(STATUS "Found lapack in ATLAS (include: ${ATLAS_CLAPACK_INC_DIR})")
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

if(OPENBLAS_LAPACKE_INC_DIR AND OPENBLAS_INC_DIR AND OPENBLAS_LIB)
  set(CBLAS_FOUND ON)
  set(CBLAS_PROVIDER OPENBLAS)
  set(CBLAS_INC_DIR ${OPENBLAS_INC_DIR} ${OPENBLAS_LAPACKE_INC_DIR})
  set(CBLAS_LIBRARIES ${OPENBLAS_LIB})

  add_definitions(-DPADDLE_USE_OPENBLAS)
  add_definitions(-DLAPACK_FOUND)

  message(STATUS "Found OpenBLAS (include: ${OPENBLAS_INC_DIR}, library: ${CBLAS_LIBRARIES})")
  message(STATUS "Found lapack in OpenBLAS (include: ${OPENBLAS_LAPACKE_INC_DIR})")
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
  set(CBLAS_FOUND ON)
  set(CBLAS_PROVIDER REFERENCE)
  set(CBLAS_INC_DIR ${REFERENCE_CBLAS_INCLUDE_DIR})
  set(CBLAS_LIBRARIES ${REFERENCE_CBLAS_LIBRARY})
  add_definitions(-DPADDLE_USE_REFERENCE_CBLAS)
  message(STATUS "Found reference-cblas (include: ${CBLAS_INC_DIR}, library: ${CBLAS_LIBRARIES})")
endif()

if(IOS_USE_VECLIB_FOR_BLAS AND VECLIB_FOUND)
  set(CBLAS_FOUND ON)
  set(CBLAS_PROVIDER vecLib)
  set(CBLAS_INC_DIR ${VECLIB_INC_DIR})
  add_definitions(-DPADDLE_USE_VECLIB)
endif()
