# Find the NNPACK library
#  NNPACK_ROOT - where to find NNPACK include and library.
#

set(NNPACK_FOUND OFF)
set(NNPACK_ROOT $ENV{NNPACK_ROOT} CACHE PATH "Folder contains NNPACK")
find_path(NNPACK_INC_DIR nnpack.h PATHS ${NNPACK_ROOT}/include)
find_library(NNPACK_LIB NAMES nnpack PATHS ${NNPACK_ROOT}/lib)
find_library(PTHREADPOOL_LIB NAMES pthreadpool PATHS ${NNPACK_ROOT}/lib)

if(NNPACK_INC_DIR AND NNPACK_LIB AND PTHREADPOOL_LIB)
  set(NNPACK_FOUND ON)
  INCLUDE_DIRECTORIES(${NNPACK_INC_DIR})
else()
  message(FATAL_ERROR "Cannot find NNPACK in (${NNPACK_ROOT})")
endif()
