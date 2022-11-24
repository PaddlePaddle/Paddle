<<<<<<< HEAD
# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
=======
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
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
# cuda moduel lazy loading is supported by CUDA 11.7+
# this experiment option makes Paddle supports lazy loading before CUDA 11.7.

<<<<<<< HEAD
option(EXP_CUDA_MODULE_LOADING_LAZY "enable lazy cuda module loading" OFF)
if(${EXP_CUDA_MODULE_LOADING_LAZY})
  if(NOT ${ON_INFER} OR NOT ${LINUX})
=======
if(LINUX)
  if(NOT ON_INFER)
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    message(
      "EXP_CUDA_MODULE_LOADING_LAZY only works with ON_INFER=ON on Linux platforms"
    )
    return()
  endif()
<<<<<<< HEAD
  if(NOT ${CUDA_FOUND})
    message("EXP_CUDA_MODULE_LOADING_LAZY only works with CUDA")
=======
  if(NOT WITH_GPU)
    message("EXP_CUDA_MODULE_LOADING_LAZY only works with GPU")
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    return()
  endif()
  if(${CUDA_VERSION} VERSION_GREATER_EQUAL "11.7")
    message("cuda 11.7+ already support lazy module loading")
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
<<<<<<< HEAD
=======
  execute_process(
    COMMAND "rm" "-rf" "${CMAKE_SOURCE_DIR}/tools/nvcc_lazy"
    COMMAND "chmod" "755" "${CMAKE_SOURCE_DIR}/tools/nvcc_lazy.sh"
    COMMAND "bash" "${CMAKE_SOURCE_DIR}/tools/nvcc_lazy.sh"
            "${CMAKE_SOURCE_DIR}/tools/nvcc_lazy" "${CUDA_TOOLKIT_ROOT_DIR}")
  execute_process(COMMAND "chmod" "755" "${CMAKE_SOURCE_DIR}/tools/nvcc_lazy")
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  set(CUDA_NVCC_EXECUTABLE
      "${CMAKE_SOURCE_DIR}/tools/nvcc_lazy"
      CACHE FILEPATH "" FORCE)
  set(CMAKE_CUDA_COMPILER
      "${CMAKE_SOURCE_DIR}/tools/nvcc_lazy"
      CACHE FILEPATH "" FORCE)
endif()
