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

if(WITH_TESTING)
    add_definitions(-DPADDLE_WITH_TESTING)
endif(WITH_TESTING)

if(NOT WITH_TIMER)
    add_definitions(-DPADDLE_DISABLE_TIMER)
endif(NOT WITH_TIMER)

if(USE_EIGEN_FOR_BLAS)
    add_definitions(-DPADDLE_USE_EIGEN_FOR_BLAS)
endif(USE_EIGEN_FOR_BLAS)

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
    add_definitions(-DHPPL_STUB_FUNC)

    list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS cu)
else()
    add_definitions(-DPADDLE_WITH_CUDA)

    FIND_PACKAGE(CUDA REQUIRED)

    if(${CUDA_VERSION_MAJOR} VERSION_LESS 7)
        message(FATAL_ERROR "Paddle needs CUDA >= 7.0 to compile")
    endif()

    if(NOT CUDNN_FOUND)
        message(FATAL_ERROR "Paddle needs cudnn to compile")
    endif()

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler ${SIMD_FLAG}")

    # Include cuda and cudnn
    include_directories(${CUDNN_INCLUDE_DIR})
    include_directories(${CUDA_TOOLKIT_INCLUDE})
endif(NOT WITH_GPU)

if(WITH_MKLDNN)
    add_definitions(-DPADDLE_USE_MKLDNN)
    if (WITH_MKLML AND MKLDNN_IOMP_DIR)
        message(STATUS "Enable Intel OpenMP at ${MKLDNN_IOMP_DIR}")
        set(OPENMP_FLAGS "-fopenmp")
        set(CMAKE_C_CREATE_SHARED_LIBRARY_FORBIDDEN_FLAGS ${OPENMP_FLAGS})
        set(CMAKE_CXX_CREATE_SHARED_LIBRARY_FORBIDDEN_FLAGS ${OPENMP_FLAGS})
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OPENMP_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENMP_FLAGS}")
    else()
        find_package(OpenMP)
        if(OPENMP_FOUND)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        else()
            message(WARNING "Can not find OpenMP."
                 "Some performance features in MKLDNN may not be available")
        endif()
    endif()

endif(WITH_MKLDNN)

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

  add_custom_target(go_path)
  add_custom_command(TARGET go_path
    # Symlink Paddle directory into GOPATH
    COMMAND mkdir -p ${PADDLE_IN_GOPATH}
    COMMAND rm -rf ${PADDLE_IN_GOPATH}
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

    # this command will only run when the file it depends is missing
    # or has changed, or the output is missing.
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/glide
      COMMAND env GOPATH=${GOPATH} ${GLIDE} install
      COMMAND touch ${CMAKE_BINARY_DIR}/glide
      DEPENDS ${PADDLE_SOURCE_DIR}/go/glide.lock
      WORKING_DIRECTORY "${PADDLE_IN_GOPATH}/go"
      )

    # depends on the custom command which outputs
    # ${CMAKE_BINARY_DIR}/glide, the custom command does not need to
    # run every time this target is built.
    add_custom_target(go_vendor DEPENDS ${CMAKE_BINARY_DIR}/glide go_path)
  endif()

endif(WITH_GOLANG)
