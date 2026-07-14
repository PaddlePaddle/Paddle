if(NOT WITH_GPU AND NOT WITH_ROCM)
  return()
endif()

include(${PROJECT_SOURCE_DIR}/cmake/architecture.cmake)

if(WITH_ROCM)
  if(EXISTS "${ROCM_PATH}/cuda/extras/CUPTI")
    set(ROCM_CUDA_DIR "${ROCM_PATH}/cuda")
  elseif(EXISTS "${ROCM_PATH}/cuda/cuda/extras/CUPTI")
    set(ROCM_CUDA_DIR "${ROCM_PATH}/cuda/cuda")
  else()
    message(
      FATAL_ERROR
        "CUPTI not found under ${ROCM_PATH}/cuda/extras/CUPTI or ${ROCM_PATH}/cuda/cuda/extras/CUPTI"
    )
  endif()
  set(CUPTI_ROOT
      "${ROCM_CUDA_DIR}/extras/CUPTI"
      CACHE PATH "CUPTI ROOT")
else()
  set(CUPTI_ROOT
      "/usr"
      CACHE PATH "CUPTI ROOT")
endif()
paddle_detect_cuda_target_dir(CUDA_TARGET_DIR)

find_path(
  CUPTI_INCLUDE_DIR cupti.h
  PATHS ${CUPTI_ROOT}
        ${CUPTI_ROOT}/include
        $ENV{CUPTI_ROOT}
        $ENV{CUPTI_ROOT}/include
        ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include
        ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include
        ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/include
        ${CUDA_TOOLKIT_ROOT_DIR}/targets/${CUDA_TARGET_DIR}/include
  NO_DEFAULT_PATH)

get_filename_component(__libpath_hist ${CUDA_CUDART_LIBRARY} PATH)

paddle_normalize_target_arch(TARGET_ARCH)
if(NOT CUDA_TARGET_DIR STREQUAL "")
  list(APPEND CUPTI_CHECK_LIBRARY_DIRS
       ${CUDA_TOOLKIT_ROOT_DIR}/targets/${CUDA_TARGET_DIR}/lib64
       ${CUDA_TOOLKIT_ROOT_DIR}/targets/${CUDA_TARGET_DIR}/lib)
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
  ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64)
find_library(
  CUPTI_LIBRARY
  NAMES libcupti.so libcupti.dylib # libcupti_static.a
  PATHS ${CUPTI_CHECK_LIBRARY_DIRS} ${CUPTI_INCLUDE_DIR} ${__libpath_hist}
  NO_DEFAULT_PATH
  DOC "Path to cuPTI library.")

get_filename_component(CUPTI_LIBRARY_PATH ${CUPTI_LIBRARY} DIRECTORY)
if(CUPTI_INCLUDE_DIR AND CUPTI_LIBRARY)
  set(CUPTI_FOUND ON)
  if(WITH_ROCM)
    include_directories(${ROCM_CUDA_DIR}/include)
    add_definitions(-D__CUDA_HIP_PLATFORM_AMD__)
  endif()
else()
  set(CUPTI_FOUND OFF)
endif()
