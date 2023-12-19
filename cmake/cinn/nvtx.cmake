if((NOT WITH_GPU)
   OR WIN32
   OR APPLE)
  set(NVTX_FOUND OFF)
  return()
endif()

set(NVTX_ROOT
    "/usr"
    CACHE PATH "NVTX ROOT")
find_path(
  NVTX_INCLUDE_DIR nvToolsExt.h
  PATHS ${NVTX_ROOT} ${NVTX_ROOT}/include $ENV{NVTX_ROOT}
        $ENV{NVTX_ROOT}/include ${CUDA_TOOLKIT_INCLUDE}
  NO_DEFAULT_PATH)

get_filename_component(__libpath_hint ${CUDA_CUDART_LIBRARY} PATH)

set(TARGET_ARCH "x86_64")
if(NOT ${CMAKE_SYSTEM_PROCESSOR})
  set(TARGET_ARCH ${CMAKE_SYSTEM_PROCESSOR})
endif()

list(
  APPEND
  NVTX_CHECK_LIBRARY_DIRS
  ${NVTX_ROOT}
  ${NVTX_ROOT}/lib64
  ${NVTX_ROOT}/lib
  ${NVTX_ROOT}/lib/${TARGET_ARCH}-linux-gnu
  $ENV{NVTX_ROOT}
  $ENV{NVTX_ROOT}/lib64
  $ENV{NVTX_ROOT}/lib
  ${CUDA_TOOLKIT_ROOT_DIR}
  ${CUDA_TOOLKIT_ROOT_DIR}/targets/${TARGET_ARCH}-linux/lib)

find_library(
  CUDA_NVTX_LIB
  NAMES libnvToolsExt.so
  PATHS ${NVTX_CHECK_LIBRARY_DIRS} ${NVTX_INCLUDE_DIR} ${__libpath_hint}
  NO_DEFAULT_PATH
  DOC "Path to the NVTX library.")

if(NVTX_INCLUDE_DIR AND CUDA_NVTX_LIB)
  set(NVTX_FOUND ON)
else()
  set(NVTX_FOUND OFF)
endif()

if(NVTX_FOUND)
  include_directories(${NVTX_INCLUDE_DIR})
  add_definitions(-DCINN_WITH_NVTX)
endif()
