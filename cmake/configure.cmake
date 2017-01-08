# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(NOT WITH_PYTHON)
    add_definitions(-DPADDLE_NO_PYTHON)
endif(NOT WITH_PYTHON)

if(WITH_DSO)
    add_definitions(-DPADDLE_USE_DSO)
endif(WITH_DSO)

if(WITH_DOUBLE)
    add_definitions(-DPADDLE_TYPE_DOUBLE)
endif(WITH_DOUBLE)

if(NOT WITH_TIMER)
    add_definitions(-DPADDLE_DISABLE_TIMER)
endif(NOT WITH_TIMER)

if(NOT WITH_PROFILER)
    add_definitions(-DPADDLE_DISABLE_PROFILER)
endif(NOT WITH_PROFILER)

if(NOT WITH_GPU)
    add_definitions(-DPADDLE_ONLY_CPU)
    add_definitions(-DHPPL_STUB_FUNC)

    list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS cu)
else()
    FIND_PACKAGE(CUDA REQUIRED)

    if(${CUDA_VERSION_MAJOR} VERSION_LESS 7)
        message(FATAL_ERROR "Paddle need CUDA >= 7.0 to compile")
    endif()

    if(NOT CUDNN_FOUND)
        message(FATAL_ERROR "Paddle need cudnn to compile")
    endif()

    if(WITH_AVX)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler ${AVX_FLAG}")
    else(WITH_AVX)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler ${SSE3_FLAG}")
    endif(WITH_AVX)

    # Include cuda and cudnn
    include_directories(${CUDNN_INCLUDE_DIR})
    include_directories(${CUDA_TOOLKIT_INCLUDE})
endif(NOT WITH_GPU)

if(WITH_AVX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${AVX_FLAG}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${AVX_FLAG}")
else(WITH_AVX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SSE3_FLAG}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE3_FLAG}")
endif(WITH_AVX)
