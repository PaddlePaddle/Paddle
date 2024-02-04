cmake_minimum_required(VERSION 3.23)
set(PYTHON_EXECUTABLE python)

#project (Demo1)
#message(${CMAKE_CURRENT_SOURCE_DIR})
#message(${CMAKE_CURRENT_BINARY_DIR})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../")

execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory
                        "${CMAKE_CURRENT_BINARY_DIR}/generated_tmp")

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/conv2d_bias_act.py"
  COMMAND ${PYTHON_EXECUTABLE}
          "${CMAKE_CURRENT_SOURCE_DIR}/conv2d_bias_residual.py"
  COMMAND ${PYTHON_EXECUTABLE}
          "${CMAKE_CURRENT_SOURCE_DIR}/conv2d_depthwise_bias_act.py"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

find_package(CUDA)

set(CUDA_NVCC_FLAGS -gencode arch=compute_75,code=sm_75;)
#set(CMAKE_CXX_FLAGS -fvisibility=hidden)
file(GLOB all_cu "${CMAKE_CURRENT_BINARY_DIR}/generated_tmp/*.cu")
list(APPEND all_cu "${CMAKE_CURRENT_SOURCE_DIR}/conv2d_util.cu")

#message(${all_cu})
cuda_add_library(CutlassConv2d SHARED ${all_cu})
