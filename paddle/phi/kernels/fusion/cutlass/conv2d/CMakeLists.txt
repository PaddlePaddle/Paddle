cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED PYTHON_EXECUTABLE)
  message(
    FATAL_ERROR
      "please set PYTHON_EXECUTABLE with -DPYTHON_EXECUTABLE=python executable path"
  )
endif()

if(NOT DEFINED COMPUTE_CAPABILITY)
  message(
    FATAL_ERROR
      "please set COMPUTE_CAPABILITY with -DCOMPUTE_CAPABILITY=your gpu compute capability"
  )
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../")

execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory
                        "${CMAKE_CURRENT_BINARY_DIR}/generated_tmp")

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/conv2d_bias_act.py
          --cuda_arch ${COMPUTE_CAPABILITY}
  COMMAND
    ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/conv2d_bias_residual.py
    --cuda_arch ${COMPUTE_CAPABILITY}
  COMMAND ${PYTHON_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/conv2d_depthwise_bias_act.py
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

find_package(CUDA)
set(CUDA_NVCC_FLAGS
    -std=c++17 -gencode
    arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY};)
#set(CMAKE_CXX_FLAGS -fvisibility=hidden)
set(CMAKE_BUILD_TYPE "Release")
file(GLOB all_cutlass_conv2d_cu
     "${CMAKE_CURRENT_BINARY_DIR}/generated_tmp/*.cu")
list(APPEND all_cutlass_conv2d_cu "${CMAKE_CURRENT_SOURCE_DIR}/conv2d_util.cu")

cuda_add_library(CutlassConv2d SHARED ${all_cutlass_conv2d_cu})
