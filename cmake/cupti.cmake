# 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.   
if(NOT WITH_GPU AND NOT WITH_ROCM)
  return()
endif()

if(WITH_ROCM)
  set(CUPTI_ROOT
      "${ROCM_PATH}/cuda/extras/CUPTI"
      CACHE PATH "CUPTI ROOT")
else()
  set(CUPTI_ROOT
      "/usr"
      CACHE PATH "CUPTI ROOT")
endif()
find_path(
  CUPTI_INCLUDE_DIR cupti.h
  PATHS ${CUPTI_ROOT}
        ${CUPTI_ROOT}/include
        $ENV{CUPTI_ROOT}
        $ENV{CUPTI_ROOT}/include
        ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include
        ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include
        ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/include
        $ENV{MACA_PATH}/tools/cu-bridge/include
  NO_DEFAULT_PATH)

get_filename_component(__libpath_hist ${CUDA_CUDART_LIBRARY} PATH)

set(TARGET_ARCH "x86_64")
if(NOT ${CMAKE_SYSTEM_PROCESSOR})
  set(TARGET_ARCH ${CMAKE_SYSTEM_PROCESSOR})
endif()

list(
  APPEND
  CUPTI_CHECK_LIBRARY_DIRS
  ${CUPTI_ROOT}
  ${CUPTI_ROOT}/lib64
  ${CUPTI_ROOT}/lib
  ${CUPTI_ROOT}/lib/${TARGET_ARCH}-linux-gnu
  $ENV{CUPTI_ROOT}
  $ENV{CUPTI_ROOT}/lib64
  $ENV{CUPTI_ROOT}/lib
  /usr/lib
  ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib64
  ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64
  $ENV{MACA_PATH}/lib)
find_library(
  CUPTI_LIBRARY
  NAMES libcupti.so libcupti.dylib libmcpti.so # libcupti_static.a
  PATHS ${CUPTI_CHECK_LIBRARY_DIRS} ${CUPTI_INCLUDE_DIR} ${__libpath_hist}
  NO_DEFAULT_PATH
  DOC "Path to cuPTI library.")

get_filename_component(CUPTI_LIBRARY_PATH ${CUPTI_LIBRARY} DIRECTORY)
if(CUPTI_INCLUDE_DIR AND CUPTI_LIBRARY)
  set(CUPTI_FOUND ON)
  if(WITH_ROCM)
    include_directories(${ROCM_PATH}/cuda/include)
    add_definitions(-D__CUDA_HIP_PLATFORM_AMD__)
  endif()
else()
  set(CUPTI_FOUND OFF)
endif()
link_libraries($ENV{MACA_PATH}/lib/libmcToolsExt.so)
