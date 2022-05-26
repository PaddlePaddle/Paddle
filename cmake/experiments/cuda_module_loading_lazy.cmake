# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
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
# cuda moduel lazy loading is supported by CUDA 11.6+
# this experiment option makes Paddle supports lazy loading before CUDA 11.6.

option(EXP_CUDA_MODULE_LOADING_LAZY  "enable lazy cuda module loading" OFF)
if (${EXP_CUDA_MODULE_LOADING_LAZY})
  if (NOT ${ON_INFER} OR NOT ${LINUX})
    message("EXP_CUDA_MODULE_LOADING_LAZY only works with ON_INFER=ON on Linux platforms")
    return()
  endif ()
  if (NOT ${CUDA_FOUND})
    message("EXP_CUDA_MODULE_LOADING_LAZY only works with CUDA")
    return()
  endif ()
  if (${CUDA_VERSION} VERSION_GREATER_EQUAL "11.6")
    message("cuda 11.6+ already support lazy module loading")
    return()
  endif ()

  message("for cuda before 11.6, libcudart.so must be used for the lazy module loading trick to work, instead of libcudart_static.a")
  set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE BOOL "" FORCE)
  set(CMAKE_CUDA_FLAGS "--cudart shared")
  enable_language(CUDA)
  set(CUDA_NVCC_EXECUTABLE "${CMAKE_SOURCE_DIR}/tools/nvcc_lazy" CACHE FILEPATH "" FORCE)
  set(CMAKE_CUDA_COMPILER "${CMAKE_SOURCE_DIR}/tools/nvcc_lazy" CACHE FILEPATH "" FORCE)
endif()

