# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# this file contains experimental build options for lazy cuda module loading
# cuda module lazy loading is supported by CUDA 11.7+
# this experiment option makes Paddle supports lazy loading before CUDA 11.7.

if(LINUX)
  if(NOT WITH_NVCC_LAZY)
    message(
      "EXP_CUDA_MODULE_LOADING_LAZY only works with WITH_NVCC_LAZY=ON on Linux platforms"
    )
    return()
  endif()
  if(NOT ON_INFER)
    message(
      "EXP_CUDA_MODULE_LOADING_LAZY only works with ON_INFER=ON on Linux platforms"
    )
    return()
  endif()
  if(NOT WITH_GPU)
    message("EXP_CUDA_MODULE_LOADING_LAZY only works with GPU")
    return()
  endif()
  if(${CUDA_VERSION} VERSION_GREATER_EQUAL "11.7")
    message("cuda 11.7+ already support lazy module loading")
    return()
  endif()
  if(${CUDA_VERSION} VERSION_LESS "12.0" AND ${CMAKE_CXX_COMPILER_VERSION}
                                             VERSION_GREATER_EQUAL 12.0)
    message("cuda less than 12.0 doesn't support gcc12")
    return()
  endif()

  message(
    "for cuda before 11.7, libcudart.so must be used for the lazy module loading trick to work, instead of libcudart_static.a"
  )
  set(CUDA_USE_STATIC_CUDA_RUNTIME
      OFF
      CACHE BOOL "" FORCE)
  set(CMAKE_CUDA_FLAGS "--cudart shared")
  enable_language(CUDA)
  set(CMAKE_CUDA_COMPILER
      "${CMAKE_SOURCE_DIR}/tools/nvcc_lazy"
      CACHE FILEPATH "" FORCE)
  execute_process(
    COMMAND "rm" "-rf" "${CMAKE_SOURCE_DIR}/tools/nvcc_lazy"
    COMMAND "chmod" "755" "${CMAKE_SOURCE_DIR}/tools/nvcc_lazy.sh"
    COMMAND "bash" "${CMAKE_SOURCE_DIR}/tools/nvcc_lazy.sh"
            "${CMAKE_SOURCE_DIR}/tools/nvcc_lazy" "${CUDA_TOOLKIT_ROOT_DIR}")
  execute_process(COMMAND "chmod" "755" "${CMAKE_SOURCE_DIR}/tools/nvcc_lazy")
  set(CUDA_NVCC_EXECUTABLE
      "${CMAKE_SOURCE_DIR}/tools/nvcc_lazy"
      CACHE FILEPATH "" FORCE)
endif()
