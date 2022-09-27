# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required(VERSION 3.8 FATAL_ERROR
)# for PyTorch extensions, version should be greater than 3.13
project(FasterTransformer LANGUAGES CXX CUDA)

find_package(CUDA 10.2 REQUIRED)

if(${CUDA_VERSION_MAJOR} VERSION_GREATER_EQUAL "11")
  add_definitions("-DENABLE_BF16")
  message(
    "CUDA_VERSION ${CUDA_VERSION_MAJOR} is greater or equal than 11, enable -DENABLE_BF16 flag"
  )
endif()

option(BUILD_TF "Build in TensorFlow mode" OFF)
option(BUILD_PYT "Build in PyTorch TorchScript class mode" OFF)
option(BUILD_TRT "Build projects about TensorRT" OFF)
if(NOT BUILD_MULTI_GPU)
  option(BUILD_MULTI_GPU "Build project about multi-GPU" OFF)
endif()
if(NOT USE_TRITONSERVER_DATATYPE)
  option(USE_TRITONSERVER_DATATYPE "Build triton backend for triton server" OFF)
endif()

option(SPARSITY_SUPPORT "Build project with Ampere sparsity feature support"
       OFF)

option(BUILD_FAST_MATH "Build in fast math mode" ON)

if(BUILD_MULTI_GPU)
  message(STATUS "Add DBUILD_MULTI_GPU, requires MPI and NCCL")
  add_definitions("-DBUILD_MULTI_GPU")
  set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
  find_package(MPI REQUIRED)
  find_package(NCCL REQUIRED)
  #if(${NCCL_VERSION} LESS 2.7)
  #  message(FATAL_ERROR "NCCL_VERSION ${NCCL_VERSION} is less than 2.7")
  #endif()
  set(CMAKE_MODULE_PATH "") # prevent the bugs for pytorch building
endif()

if(BUILD_PYT)
  if(DEFINED ENV{NVIDIA_PYTORCH_VERSION})
    if($ENV{NVIDIA_PYTORCH_VERSION} VERSION_LESS "20.03")
      message(
        FATAL_ERROR "NVIDIA PyTorch image is too old for TorchScript mode.")
    endif()
    if($ENV{NVIDIA_PYTORCH_VERSION} VERSION_EQUAL "20.03")
      add_definitions(-DLEGACY_THS=1)
    endif()
  endif()
endif()

if(USE_TRITONSERVER_DATATYPE)
  message("-- USE_TRITONSERVER_DATATYPE")
  add_definitions("-DUSE_TRITONSERVER_DATATYPE")
endif()

set(CXX_STD
    "14"
    CACHE STRING "C++ standard")

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

set(TF_PATH
    ""
    CACHE STRING "TensorFlow path")
set(CUSPARSELT_PATH
    ""
    CACHE STRING "cuSPARSELt path")

if(BUILD_TF AND NOT TF_PATH)
  message(
    FATAL_ERROR "TF_PATH must be set if BUILD_TF(=TensorFlow mode) is on.")
endif()

list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

# profiling
option(USE_NVTX "Whether or not to use nvtx" OFF)
if(USE_NVTX)
  message(STATUS "NVTX is enabled.")
  add_definitions("-DUSE_NVTX")
endif()

# setting compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall -ldl")

set(SM_SETS
    52
    60
    61
    70
    75
    80
    86)
set(USING_WMMA False)
set(FIND_SM False)

foreach(SM_NUM IN LISTS SM_SETS)
  string(FIND "${SM}" "${SM_NUM}" SM_POS)
  if(SM_POS GREATER -1)
    if(FIND_SM STREQUAL False)
      set(ENV{TORCH_CUDA_ARCH_LIST} "")
    endif()
    set(FIND_SM True)
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${SM_NUM},code=\\\"sm_${SM_NUM},compute_${SM_NUM}\\\""
    )

    if(SM_NUM STREQUAL 70
       OR SM_NUM STREQUAL 75
       OR SM_NUM STREQUAL 80
       OR SM_NUM STREQUAL 86)
      set(USING_WMMA True)
    endif()

    if(BUILD_PYT)
      string(SUBSTRING ${SM_NUM} 0 1 SM_MAJOR)
      string(SUBSTRING ${SM_NUM} 1 1 SM_MINOR)
      set(ENV{TORCH_CUDA_ARCH_LIST}
          "$ENV{TORCH_CUDA_ARCH_LIST}\;${SM_MAJOR}.${SM_MINOR}")
    endif()

    set(CMAKE_CUDA_ARCHITECTURES ${SM_NUM})
    message("-- Assign GPU architecture (sm=${SM_NUM})")
  endif()
endforeach()

if(USING_WMMA STREQUAL True)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}    -DWMMA")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -DWMMA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
  message("-- Use WMMA")
endif()

if(NOT (FIND_SM STREQUAL True))
  set(CMAKE_CUDA_FLAGS
      "${CMAKE_CUDA_FLAGS}  \
                        -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                        -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                        -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                        -gencode=arch=compute_86,code=\\\"sm_86,compute_86\\\" \
                        ")
  #                      -rdc=true")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}    -DWMMA")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -DWMMA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
  if(BUILD_PYT)
    set(ENV{TORCH_CUDA_ARCH_LIST} "7.0;7.5;8.0;8.6")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
  message("-- Assign GPU architecture (sm=70,75,80,86)")
endif()

if(BUILD_PYT)
  set(TORCH_CUDA_ARCH_LIST $ENV{TORCH_CUDA_ARCH_LIST})
endif()

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall  --ptxas-options=-v --resource-usage")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")

set(CMAKE_CXX_STANDARD "${CXX_STD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++${CXX_STD}")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
# set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3 --ptxas-options=--verbose")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3")
if(BUILD_FAST_MATH)
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math")
  message("CMAKE_CUDA_FLAGS_RELEASE: ${CMAKE_CUDA_FLAGS_RELEASE}")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(COMMON_HEADER_DIRS ${PROJECT_SOURCE_DIR} ${CUDA_PATH}/include)
message("-- COMMON_HEADER_DIRS: ${COMMON_HEADER_DIRS}")

set(COMMON_LIB_DIRS ${CUDA_PATH}/lib64)

if(SPARSITY_SUPPORT)
  list(APPEND COMMON_HEADER_DIRS ${CUSPARSELT_PATH}/include)
  list(APPEND COMMON_LIB_DIRS ${CUSPARSELT_PATH}/lib64)
  add_definitions(-DSPARSITY_ENABLED=1)
endif()

if(BUILD_TF)
  list(APPEND COMMON_HEADER_DIRS ${TF_PATH}/include)
  list(APPEND COMMON_LIB_DIRS ${TF_PATH})
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()

set(PYTHON_PATH
    "python"
    CACHE STRING "Python path")
if(BUILD_PYT)
  execute_process(
    COMMAND
      ${PYTHON_PATH} "-c"
      "from __future__ import print_function; import torch; print(torch.__version__,end='');"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE TORCH_VERSION)
  if(TORCH_VERSION VERSION_LESS "1.5.0")
    message(FATAL_ERROR "PyTorch >= 1.5.0 is needed for TorchScript mode.")
  endif()
  execute_process(
    COMMAND ${PYTHON_PATH} "-c"
            "from __future__ import print_function; import os; import torch;
print(os.path.dirname(torch.__file__),end='');"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE TORCH_DIR)
  if(NOT _PYTHON_SUCCESS MATCHES 0)
    message(FATAL_ERROR "Torch config Error.")
  endif()
  list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
  find_package(Torch REQUIRED)
  execute_process(
    COMMAND
      ${PYTHON_PATH} "-c"
      "from __future__ import print_function; from distutils import sysconfig;
print(sysconfig.get_python_inc());"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE PY_INCLUDE_DIR)
  if(NOT _PYTHON_SUCCESS MATCHES 0)
    message(FATAL_ERROR "Python config Error.")
  endif()
  list(APPEND COMMON_HEADER_DIRS ${PY_INCLUDE_DIR})
  execute_process(
    COMMAND ${PYTHON_PATH} "-c"
            "from __future__ import print_function; import torch;
print(torch._C._GLIBCXX_USE_CXX11_ABI,end='');"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE USE_CXX11_ABI)
  message("-- USE_CXX11_ABI=${USE_CXX11_ABI}")
  if(USE_CXX11_ABI)
    set(CMAKE_CUDA_FLAGS_RELEASE
        "${CMAKE_CUDA_FLAGS_RELEASE} -D_GLIBCXX_USE_CXX11_ABI=1")
    set(CMAKE_CXX_FLAGS_RELEASE
        "${CMAKE_CXX_FLAGS_RELEASE} -D_GLIBCXX_USE_CXX11_ABI=1")
    set(CMAKE_CUDA_FLAGS_DEBUG
        "${CMAKE_CUDA_FLAGS_DEBUG} -D_GLIBCXX_USE_CXX11_ABI=1")
    set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} -D_GLIBCXX_USE_CXX11_ABI=1")
  else()
    set(CMAKE_CUDA_FLAGS_RELEASE
        "${CMAKE_CUDA_FLAGS_RELEASE} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CXX_FLAGS_RELEASE
        "${CMAKE_CXX_FLAGS_RELEASE} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CUDA_FLAGS_DEBUG
        "${CMAKE_CUDA_FLAGS_DEBUG} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} -D_GLIBCXX_USE_CXX11_ABI=0")
  endif()
endif()

if(BUILD_MULTI_GPU)
  list(APPEND COMMON_HEADER_DIRS ${MPI_INCLUDE_PATH})
  list(APPEND COMMON_LIB_DIRS /usr/local/mpi/lib)
endif()

if(USE_TRITONSERVER_DATATYPE)
  list(APPEND COMMON_HEADER_DIRS ${PROJECT_SOURCE_DIR}/../repo-core-src/include)
endif()

include_directories(${COMMON_HEADER_DIRS})

link_directories(${COMMON_LIB_DIRS})

add_subdirectory(3rdparty)

# # Mesaure the compile time
option(MEASURE_BUILD_TIME "Measure the build time of each module" OFF)
if(MEASURE_BUILD_TIME)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_CUSTOM "${CMAKE_COMMAND} -E time")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")
endif()

########################################

# Following feature requires cmake 3.15
# TODO Remove this part or modify such that we can run it under cmake 3.10
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
add_library(transformer-static STATIC
            $<TARGET_OBJECTS:trt_fused_multi_head_attention>)
set_property(TARGET transformer-static PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET transformer-static PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(transformer-static PUBLIC -lcudart -lcublas -lcublasLt
                                                -lcurand)

add_library(transformer-shared SHARED
            $<TARGET_OBJECTS:trt_fused_multi_head_attention>)
set_target_properties(transformer-shared PROPERTIES POSITION_INDEPENDENT_CODE
                                                    ON)
set_target_properties(transformer-shared PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS
                                                    ON)
set_target_properties(transformer-shared PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(transformer-shared PUBLIC -lcudart -lcublas -lcublasLt
                                                -lcurand)

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/FasterTransformer)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/FasterTransformerConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/FasterTransformerConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR})

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/FasterTransformerConfig.cmake
        DESTINATION ${INSTALL_CONFIGDIR})

install(
  TARGETS transformer-shared
  EXPORT transformer-shared-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/fastertransformer
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/fastertransformer)

install(
  EXPORT transformer-shared-targets
  FILE FasterTransformerTargets.cmake
  DESTINATION ${INSTALL_CONFIGDIR})

file(GLOB_RECURSE HEADER_FILES "*.h" "*.hpp" "*.cuh")
foreach(file ${HEADER_FILES})
  file(RELATIVE_PATH rfile ${CMAKE_CURRENT_SOURCE_DIR} ${file})
  get_filename_component(dir ${rfile} DIRECTORY)
  install(FILES ${file} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${dir})
endforeach()

################################################################################
# add_executable(gpt sample/cpp/gpt_sample.cc )
# target_link_libraries(gpt PUBLIC -lcublas -lcublasLt -lcudart -lcurand -lnccl -lmpi transformer-static)
# target_link_libraries(gpt PUBLIC -lcublas -lcublasLt -lcudart -lcurand -lnccl -lmpi decoder decoding)

export(
  EXPORT transformer-shared-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/FasterTransformerTargets.cmake
  NAMESPACE TritonCore::)

export(PACKAGE FasterTransformer)
