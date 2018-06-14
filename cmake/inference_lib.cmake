# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

set_property(GLOBAL PROPERTY FLUID_MODULES "")
# find all fluid modules is used for paddle fluid static library
function(find_fluid_modules TARGET_NAME)
  get_filename_component(__target_path ${TARGET_NAME} ABSOLUTE)
  string(REGEX REPLACE "^${PADDLE_SOURCE_DIR}/" "" __target_path ${__target_path})
  string(FIND "${__target_path}" "fluid" pos)
  if(pos GREATER 1)
    get_property(fluid_modules GLOBAL PROPERTY FLUID_MODULES)
    set(fluid_modules ${fluid_modules} ${TARGET_NAME})
    set_property(GLOBAL PROPERTY FLUID_MODULES "${fluid_modules}")
  endif()
endfunction(find_fluid_modules)

# make package for paddle fluid shared and static library
function(copy TARGET)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DSTS DEPS)
    cmake_parse_arguments(copy_lib "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set(inference_lib_dist_dep ${TARGET} ${inference_lib_dist_dep} PARENT_SCOPE)

    list(LENGTH copy_lib_SRCS copy_lib_SRCS_len)
    list(LENGTH copy_lib_DSTS copy_lib_DSTS_len)
    if(NOT ${copy_lib_SRCS_len} EQUAL ${copy_lib_DSTS_len})
        message(FATAL_ERROR "${TARGET} source numbers are not equal to destination numbers")
    endif()
    math(EXPR len "${copy_lib_SRCS_len} - 1")

    add_custom_target(${TARGET} DEPENDS ${copy_lib_DEPS})
    foreach(index RANGE ${len})
        list(GET copy_lib_SRCS ${index} src)
        list(GET copy_lib_DSTS ${index} dst)
        add_custom_command(TARGET ${TARGET} PRE_BUILD 
          COMMAND mkdir -p "${dst}"
          COMMAND cp -r "${src}" "${dst}"
          COMMENT "copying ${src} -> ${dst}")
    endforeach()
endfunction()

# third party
set(dst_dir "${FLUID_INSTALL_DIR}/third_party/eigen3")
copy(eigen3_lib
  SRCS ${EIGEN_INCLUDE_DIR}/Eigen/Core ${EIGEN_INCLUDE_DIR}/Eigen/src ${EIGEN_INCLUDE_DIR}/unsupported/Eigen
  DSTS ${dst_dir}/Eigen ${dst_dir}/Eigen ${dst_dir}/unsupported
  DEPS eigen3
)

set(dst_dir "${FLUID_INSTALL_DIR}/third_party/install/gflags")
copy(gflags_lib
  SRCS ${GFLAGS_INCLUDE_DIR} ${GFLAGS_LIBRARIES}
  DSTS ${dst_dir} ${dst_dir}/lib
  DEPS gflags
)

set(dst_dir "${FLUID_INSTALL_DIR}/third_party/install/glog")
copy(glog_lib
  SRCS ${GLOG_INCLUDE_DIR} ${GLOG_LIBRARIES}
  DSTS ${dst_dir} ${dst_dir}/lib
  DEPS glog
)

set(dst_dir "${FLUID_INSTALL_DIR}/third_party/boost/")
copy(boost_lib
  SRCS ${BOOST_INCLUDE_DIR}/boost
  DSTS ${dst_dir}
  DEPS boost
)

if(NOT PROTOBUF_FOUND)
    set(dst_dir "${FLUID_INSTALL_DIR}/third_party/install/protobuf")
    copy(protobuf_lib
      SRCS ${PROTOBUF_INCLUDE_DIR} ${PROTOBUF_LIBRARY}
      DSTS ${dst_dir} ${dst_dir}/lib
      DEPS extern_protobuf
    )
endif()

if(NOT CBLAS_FOUND)
    set(dst_dir "${FLUID_INSTALL_DIR}/third_party/install/openblas")
    copy(openblas_lib
      SRCS ${CBLAS_INSTALL_DIR}/lib ${CBLAS_INSTALL_DIR}/include
      DSTS ${dst_dir} ${dst_dir}
      DEPS extern_openblas
    )
elseif (WITH_MKLML)
    set(dst_dir "${FLUID_INSTALL_DIR}/third_party/install/mklml")
    copy(mklml_lib
      SRCS ${MKLML_LIB} ${MKLML_IOMP_LIB} ${MKLML_INC_DIR}
      DSTS ${dst_dir}/lib ${dst_dir}/lib ${dst_dir}
      DEPS mklml
    )
endif()

if(WITH_MKLDNN)
  set(dst_dir "${FLUID_INSTALL_DIR}/third_party/install/mkldnn")
  copy(mkldnn_lib
    SRCS ${MKLDNN_INC_DIR} ${MKLDNN_SHARED_LIB}
    DSTS ${dst_dir} ${dst_dir}/lib
    DEPS mkldnn
  )
endif()

if(NOT MOBILE_INFERENCE AND NOT RPI)
  set(dst_dir "${FLUID_INSTALL_DIR}/third_party/install/snappy")
  copy(snappy_lib
    SRCS ${SNAPPY_INCLUDE_DIR} ${SNAPPY_LIBRARIES}
    DSTS ${dst_dir} ${dst_dir}/lib
    DEPS snappy)

  set(dst_dir "${FLUID_INSTALL_DIR}/third_party/install/snappystream")
  copy(snappystream_lib
    SRCS ${SNAPPYSTREAM_INCLUDE_DIR} ${SNAPPYSTREAM_LIBRARIES}
    DSTS ${dst_dir} ${dst_dir}/lib
    DEPS snappystream)

  set(dst_dir "${FLUID_INSTALL_DIR}/third_party/install/zlib")
  copy(zlib_lib
    SRCS ${ZLIB_INCLUDE_DIR} ${ZLIB_LIBRARIES}
    DSTS ${dst_dir} ${dst_dir}/lib
    DEPS zlib)
endif()

# paddle fluid module
set(src_dir "${PADDLE_SOURCE_DIR}/paddle/fluid")
set(dst_dir "${FLUID_INSTALL_DIR}/paddle/fluid")
set(module "framework")
copy(framework_lib DEPS framework_py_proto 
  SRCS ${src_dir}/${module}/*.h ${src_dir}/${module}/details/*.h ${PADDLE_BINARY_DIR}/paddle/fluid/framework/framework.pb.h
  DSTS ${dst_dir}/${module} ${dst_dir}/${module}/details ${dst_dir}/${module}
)

set(module "memory")
copy(memory_lib
  SRCS ${src_dir}/${module}/*.h ${src_dir}/${module}/detail/*.h
  DSTS ${dst_dir}/${module} ${dst_dir}/${module}/detail
)

set(module "inference")
copy(inference_lib DEPS paddle_fluid_shared paddle_fluid
  SRCS ${src_dir}/${module}/*.h ${PADDLE_BINARY_DIR}/paddle/fluid/inference/libpaddle_fluid.*
  DSTS ${dst_dir}/${module} ${dst_dir}/${module}
)

if(WITH_CONTRIB)
   set(contrib_dst_dir "${FLUID_INSTALL_DIR}/contrib/inference")
   copy(contrib_inference_lib DEPS paddle_inference_api
        SRCS ${PADDLE_SOURCE_DIR}/paddle/contrib/inference/paddle_inference_api.h
        ${PADDLE_BINARY_DIR}/paddle/contrib/inference/libpaddle_inference_api.*
        DSTS ${contrib_dst_dir} ${contrib_dst_dir}
   )
endif()

set(module "platform")
copy(platform_lib DEPS profiler_py_proto
  SRCS ${src_dir}/${module}/*.h ${src_dir}/${module}/dynload/*.h ${src_dir}/${module}/details/*.h
  DSTS ${dst_dir}/${module} ${dst_dir}/${module}/dynload ${dst_dir}/${module}/details
)

set(module "string")
copy(string_lib
  SRCS ${src_dir}/${module}/*.h ${src_dir}/${module}/tinyformat/*.h
  DSTS ${dst_dir}/${module} ${dst_dir}/${module}/tinyformat
)

set(module "pybind")
copy(pybind_lib
  SRCS ${CMAKE_CURRENT_BINARY_DIR}/paddle/fluid/${module}/pybind.h
  DSTS ${dst_dir}/${module}
)

# CMakeCache Info
copy(cmake_cache
  SRCS ${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt
  DSTS ${FLUID_INSTALL_DIR})

add_custom_target(inference_lib_dist DEPENDS ${inference_lib_dist_dep}) 

# paddle fluid version
execute_process(
  COMMAND ${GIT_EXECUTABLE} log --pretty=format:%H -1
  WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}
  OUTPUT_VARIABLE PADDLE_GIT_COMMIT)
set(version_file ${FLUID_INSTALL_DIR}/version.txt)
file(WRITE ${version_file}
  "GIT COMMIT ID: ${PADDLE_GIT_COMMIT}\n"
  "WITH_MKL: ${WITH_MKL}\n"
  "WITH_GPU: ${WITH_GPU}\n")
if(WITH_GPU)
  file(APPEND ${version_file}
    "CUDA version: ${CUDA_VERSION}\n"
    "CUDNN version: v${CUDNN_MAJOR_VERSION}\n")
endif()
