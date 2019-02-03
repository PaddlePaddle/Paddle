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

if(NOT WITH_PYTHON)
    add_definitions(-DPADDLE_NO_PYTHON)
endif(NOT WITH_PYTHON)

if(WITH_DSO)
    add_definitions(-DPADDLE_USE_DSO)
endif(WITH_DSO)

if(WITH_DOUBLE)
    add_definitions(-DPADDLE_TYPE_DOUBLE)
endif(WITH_DOUBLE)

if(WITH_ARM_FP16)
    add_definitions(-DPADDLE_ARM_FP16)
    add_definitions("-march=armv8.2-a+fp16+simd")
endif(WITH_ARM_FP16)

if(WITH_TESTING)
    add_definitions(-DPADDLE_WITH_TESTING)
endif(WITH_TESTING)

if(NOT WITH_TIMER)
    add_definitions(-DPADDLE_DISABLE_TIMER)
endif(NOT WITH_TIMER)

if(USE_EIGEN_FOR_BLAS)
    add_definitions(-DPADDLE_USE_EIGEN_FOR_BLAS)
endif(USE_EIGEN_FOR_BLAS)

if(EIGEN_USE_THREADS)
    add_definitions(-DEIGEN_USE_THREADS)
endif(EIGEN_USE_THREADS)

if(NOT WITH_PROFILER)
    add_definitions(-DPADDLE_DISABLE_PROFILER)
endif(NOT WITH_PROFILER)

if(WITH_AVX AND AVX_FOUND)
    set(SIMD_FLAG ${AVX_FLAG})
elseif(SSE3_FOUND)
    set(SIMD_FLAG ${SSE3_FLAG})
endif()

if(WIN32)
  # windows header option for all targets.
  add_definitions(-D_XKEYCHECK_H)
  # Use symbols instead of absolute path, reduce the cmake link command length. 
  SET(CMAKE_C_USE_RESPONSE_FILE_FOR_LIBRARIES 1)
  SET(CMAKE_CXX_USE_RESPONSE_FILE_FOR_LIBRARIES 1)
  SET(CMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS 1)
  SET(CMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS 1)
  SET(CMAKE_C_USE_RESPONSE_FILE_FOR_INCLUDES 1)
  SET(CMAKE_CXX_USE_RESPONSE_FILE_FOR_INCLUDES 1)
  SET(CMAKE_C_RESPONSE_FILE_LINK_FLAG "@")
  SET(CMAKE_CXX_RESPONSE_FILE_LINK_FLAG "@")

  # Specify the program to use when building static libraries
  SET(CMAKE_C_CREATE_STATIC_LIBRARY "<CMAKE_AR> lib <TARGET> <LINK_FLAGS> <OBJECTS>")
  SET(CMAKE_CXX_CREATE_STATIC_LIBRARY "<CMAKE_AR> lib <TARGET> <LINK_FLAGS> <OBJECTS>")

  # set defination for the dll export
  if (NOT MSVC)
    message(FATAL "Windows build only support msvc. Which was binded by the nvcc compiler of NVIDIA.")
  endif(NOT MSVC)
endif(WIN32)

if(NOT WITH_GOLANG)
    add_definitions(-DPADDLE_WITHOUT_GOLANG)
endif(NOT WITH_GOLANG)

if(WITH_PSLIB)
    add_definitions(-DPADDLE_WITH_PSLIB)
endif()

if(WITH_GPU)
    add_definitions(-DPADDLE_WITH_CUDA)
    add_definitions(-DEIGEN_USE_GPU)

    FIND_PACKAGE(CUDA REQUIRED)

    if(${CUDA_VERSION_MAJOR} VERSION_LESS 7)
        message(FATAL_ERROR "Paddle needs CUDA >= 7.0 to compile")
    endif()

    if(NOT CUDNN_FOUND)
        message(FATAL_ERROR "Paddle needs cudnn to compile")
    endif()
    if(CUPTI_FOUND)
        include_directories(${CUPTI_INCLUDE_DIR})
        add_definitions(-DPADDLE_WITH_CUPTI)
    else()
        message(STATUS "Cannot find CUPTI, GPU Profiling is incorrect.")
    endif()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler ${SIMD_FLAG}")

    # Include cuda and cudnn
    include_directories(${CUDNN_INCLUDE_DIR})
    include_directories(${CUDA_TOOLKIT_INCLUDE})

    if(TENSORRT_FOUND)
        if(${CUDA_VERSION_MAJOR} VERSION_LESS 8)
            message(FATAL_ERROR "TensorRT needs CUDA >= 8.0 to compile")
        endif()
        if(${CUDNN_MAJOR_VERSION} VERSION_LESS 7)
            message(FATAL_ERROR "TensorRT needs CUDNN >= 7.0 to compile")
        endif()
        if(${TENSORRT_MAJOR_VERSION} VERSION_LESS 4)
            message(FATAL_ERROR "Paddle needs TensorRT >= 4.0 to compile")
        endif()
        include_directories(${TENSORRT_INCLUDE_DIR})
    endif()
    if(WITH_ANAKIN)
        if(${CUDA_VERSION_MAJOR} VERSION_LESS 8)
            message(WARNING "Anakin needs CUDA >= 8.0 to compile. Force WITH_ANAKIN=OFF")
            set(WITH_ANAKIN OFF CACHE STRING "Anakin is valid only when CUDA >= 8.0." FORCE)
        endif()
        if(${CUDNN_MAJOR_VERSION} VERSION_LESS 7)
            message(WARNING "Anakin needs CUDNN >= 7.0 to compile. Force WITH_ANAKIN=OFF")
            set(WITH_ANAKIN OFF CACHE STRING "Anakin is valid only when CUDNN >= 7.0." FORCE)
        endif()
        add_definitions(-DWITH_ANAKIN)
    endif()
    if(WITH_ANAKIN)
        # NOTICE(minqiyang): the end slash is important because $CUDNN_INCLUDE_DIR
        # is a softlink to real cudnn.h directory
        set(ENV{CUDNN_INCLUDE_DIR} "${CUDNN_INCLUDE_DIR}/")
        get_filename_component(CUDNN_LIBRARY_DIR ${CUDNN_LIBRARY} DIRECTORY)
        set(ENV{CUDNN_LIBRARY} ${CUDNN_LIBRARY_DIR})
    endif()
elseif(WITH_AMD_GPU)
    add_definitions(-DPADDLE_WITH_HIP)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__HIP_PLATFORM_HCC__")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_PLATFORM_HCC__")
else()
    add_definitions(-DHPPL_STUB_FUNC)
    list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS cu)
endif()

if (WITH_MKLML AND MKLML_IOMP_LIB)
    message(STATUS "Enable Intel OpenMP with ${MKLML_IOMP_LIB}")
    if(WIN32)
        # openmp not support well for now on windows
        set(OPENMP_FLAGS "")
    else(WIN32)
        set(OPENMP_FLAGS "-fopenmp")
    endif(WIN32)
    set(CMAKE_C_CREATE_SHARED_LIBRARY_FORBIDDEN_FLAGS ${OPENMP_FLAGS})
    set(CMAKE_CXX_CREATE_SHARED_LIBRARY_FORBIDDEN_FLAGS ${OPENMP_FLAGS})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OPENMP_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENMP_FLAGS}")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SIMD_FLAG}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SIMD_FLAG}")

if(WITH_DISTRIBUTE)
  add_definitions(-DPADDLE_WITH_DISTRIBUTE)
endif()

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

if(WITH_GRPC)
    add_definitions(-DPADDLE_WITH_GRPC)
endif(WITH_GRPC)

if(WITH_BRPC_RDMA)
    add_definitions(-DPADDLE_WITH_BRPC_RDMA)
endif(WITH_BRPC_RDMA)

if(ON_INFER)
    add_definitions(-DPADDLE_ON_INFERENCE)
endif(ON_INFER)
