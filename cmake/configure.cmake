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

if(NOT CMAKE_CROSSCOMPILING)
    if(WITH_AVX AND AVX_FOUND)
        set(SIMD_FLAG ${AVX_FLAG})
    elseif(SSE3_FOUND)
        set(SIMD_FLAG ${SSE3_FLAG})
    endif()
endif()

if(NOT WITH_GOLANG)
    add_definitions(-DPADDLE_WITHOUT_GOLANG)
endif(NOT WITH_GOLANG)

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

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler ${SIMD_FLAG}")

    # Include cuda and cudnn
    include_directories(${CUDNN_INCLUDE_DIR})
    include_directories(${CUDA_TOOLKIT_INCLUDE})
endif(NOT WITH_GPU)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SIMD_FLAG}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SIMD_FLAG}")

if(WITH_GOLANG)
  # we need to symlink Paddle directory into GOPATH. If we
  # don't do it and we have code that depends on Paddle, go
  # get ./... will download a new Paddle repo from Github,
  # without the changes in our current Paddle repo that we
  # want to build.
  set(GOPATH "${CMAKE_CURRENT_BINARY_DIR}/go")
  file(MAKE_DIRECTORY ${GOPATH})
  set(PADDLE_IN_GOPATH "${GOPATH}/src/github.com/PaddlePaddle/Paddle")
  file(MAKE_DIRECTORY "${PADDLE_IN_GOPATH}")
  set(PADDLE_GO_PATH "${CMAKE_SOURCE_DIR}/go")

  message(STATUS "link ${CMAKE_SOURCE_DIR} to ${PADDLE_IN_GOPATH}...")

  execute_process(
    # Symlink Paddle directory into GOPATH
    COMMAND mkdir -p ${PADDLE_IN_GOPATH}
    COMMAND rm -rf ${PADDLE_IN_GOPATH}/paddle
    COMMAND ln -sf ${CMAKE_SOURCE_DIR} ${PADDLE_IN_GOPATH}
    # Automatically get all dependencies specified in the source code
    # We can't run `go get -d ./...` for every target, because
    # multiple `go get` can not run concurrently, but make need to be
    # able to run with multiple jobs.
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  if (GLIDE_INSTALL)
    if(EXISTS $ENV{GOPATH}/bin/glide)
      set(GLIDE "$ENV{GOPATH}/bin/glide")
    else()
      message(FATAL_ERROR "no glide executeble found: $ENV{GOPATH}/bin/glide")
    endif()

    message(STATUS "glide installing...")

    execute_process(
      COMMAND env GOPATH=${GOPATH} ${GLIDE} install
      WORKING_DIRECTORY "${PADDLE_IN_GOPATH}/paddle/go"
      RESULT_VARIABLE res_var
    )
    if(NOT "${res_var}" STREQUAL "0")
      message(FATAL_ERROR "glide install error")
    endif()

  endif()

endif(WITH_GOLANG)
