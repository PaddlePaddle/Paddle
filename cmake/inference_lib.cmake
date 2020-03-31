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

# make package for paddle fluid shared and static library
set(FLUID_INSTALL_DIR "${CMAKE_BINARY_DIR}/fluid_install_dir" CACHE STRING
  "A path setting fluid shared and static libraries")

set(FLUID_INFERENCE_INSTALL_DIR "${CMAKE_BINARY_DIR}/fluid_inference_install_dir" CACHE STRING
  "A path setting fluid inference shared and static libraries")
  
if(WIN32)
    if(NOT PYTHON_EXECUTABLE)
        FIND_PACKAGE(PythonInterp REQUIRED)
    endif()
endif()

set(COPY_SCRIPT_DIR ${PADDLE_SOURCE_DIR}/cmake)
function(copy TARGET)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DSTS)
    cmake_parse_arguments(copy_lib "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    list(LENGTH copy_lib_SRCS copy_lib_SRCS_len)
    list(LENGTH copy_lib_DSTS copy_lib_DSTS_len)
    if (NOT ${copy_lib_SRCS_len} EQUAL ${copy_lib_DSTS_len})
        message(FATAL_ERROR "${TARGET} source numbers are not equal to destination numbers")
    endif ()
    math(EXPR len "${copy_lib_SRCS_len} - 1")
    foreach (index RANGE ${len})
        list(GET copy_lib_SRCS ${index} src)
        list(GET copy_lib_DSTS ${index} dst)
        if (WIN32)   #windows
            file(TO_NATIVE_PATH ${src} native_src)
            file(TO_NATIVE_PATH ${dst} native_dst)
            add_custom_command(TARGET ${TARGET} POST_BUILD
                    COMMAND ${PYTHON_EXECUTABLE} ${COPY_SCRIPT_DIR}/copyfile.py ${native_src} ${native_dst})
        else (WIN32) #not windows
            add_custom_command(TARGET ${TARGET} POST_BUILD
                    COMMAND mkdir -p "${dst}"
                    COMMAND cp -r "${src}" "${dst}"
                    COMMENT "copying ${src} -> ${dst}")
        endif (WIN32) # not windows
    endforeach ()
endfunction()

function(copy_part_of_thrid_party TARGET DST) 
    if(${CBLAS_PROVIDER} STREQUAL MKLML)
        set(dst_dir "${DST}/third_party/install/mklml")
        if(WIN32)
            copy(${TARGET}
                    SRCS ${MKLML_LIB} ${MKLML_IOMP_LIB} ${MKLML_SHARED_LIB}
                    ${MKLML_SHARED_LIB_DEPS} ${MKLML_SHARED_IOMP_LIB} ${MKLML_INC_DIR}
                    DSTS ${dst_dir}/lib ${dst_dir}/lib ${dst_dir}/lib
                    ${dst_dir}/lib ${dst_dir}/lib ${dst_dir})
        else()
            copy(${TARGET}
                    SRCS ${MKLML_LIB} ${MKLML_IOMP_LIB} ${MKLML_INC_DIR}
                    DSTS ${dst_dir}/lib ${dst_dir}/lib ${dst_dir})
        endif()
    elseif(${CBLAS_PROVIDER} STREQUAL EXTERN_OPENBLAS)
        set(dst_dir "${DST}/third_party/install/openblas")
	if(WIN32)
            copy(${TARGET}
                    SRCS ${CBLAS_INSTALL_DIR}/lib ${OPENBLAS_SHARED_LIB} ${CBLAS_INSTALL_DIR}/include
                    DSTS ${dst_dir} ${dst_dir}/lib ${dst_dir})
	else()
            copy(${TARGET}
                    SRCS ${CBLAS_INSTALL_DIR}/lib ${CBLAS_INSTALL_DIR}/include
                    DSTS ${dst_dir} ${dst_dir})
	endif()
    endif()

    if(WITH_MKLDNN)
        set(dst_dir "${DST}/third_party/install/mkldnn")
        if(WIN32)
            copy(${TARGET}
                    SRCS ${MKLDNN_INC_DIR} ${MKLDNN_SHARED_LIB}  ${MKLDNN_LIB}
                    DSTS ${dst_dir} ${dst_dir}/lib ${dst_dir}/lib)
        else()
            copy(${TARGET}
                    SRCS ${MKLDNN_INC_DIR} ${MKLDNN_SHARED_LIB} ${MKLDNN_SHARED_LIB_1}
                    DSTS ${dst_dir} ${dst_dir}/lib ${dst_dir}/lib)
        endif()
    endif()

    set(dst_dir "${DST}/third_party/install/gflags")
    copy(${TARGET}
            SRCS ${GFLAGS_INCLUDE_DIR} ${GFLAGS_LIBRARIES}
            DSTS ${dst_dir} ${dst_dir}/lib)

    set(dst_dir "${DST}/third_party/install/glog")
    copy(${TARGET}
            SRCS ${GLOG_INCLUDE_DIR} ${GLOG_LIBRARIES}
            DSTS ${dst_dir} ${dst_dir}/lib)

    set(dst_dir "${DST}/third_party/install/xxhash")
    copy(${TARGET}
        SRCS ${XXHASH_INCLUDE_DIR} ${XXHASH_LIBRARIES}
        DSTS ${dst_dir} ${dst_dir}/lib)    
            
    if (NOT PROTOBUF_FOUND OR WIN32)
        set(dst_dir "${DST}/third_party/install/protobuf")
        copy(${TARGET}
                SRCS ${PROTOBUF_INCLUDE_DIR} ${PROTOBUF_LIBRARY}
                DSTS ${dst_dir} ${dst_dir}/lib)
    endif ()

    if (WITH_NGRAPH)
        set(dst_dir "${DST}/third_party/install/ngraph")
        copy(${TARGET}
                SRCS ${NGRAPH_INC_DIR} ${NGRAPH_LIB_DIR}
                DSTS ${dst_dir} ${dst_dir})
    endif ()

    if (LITE_BINARY_DIR)
        set(dst_dir "${DST}/third_party/install/lite")
        copy(${TARGET}
                SRCS ${LITE_BINARY_DIR}/inference_lite_lib/*
                DSTS ${dst_dir})
    endif()
endfunction()

# inference library for only inference
set(inference_lib_deps third_party paddle_fluid paddle_fluid_c paddle_fluid_shared paddle_fluid_c_shared)
add_custom_target(inference_lib_dist DEPENDS ${inference_lib_deps})


set(dst_dir "${FLUID_INFERENCE_INSTALL_DIR}/third_party/threadpool")
copy(inference_lib_dist
        SRCS ${THREADPOOL_INCLUDE_DIR}/ThreadPool.h
        DSTS ${dst_dir})

copy(inference_lib_dist
        SRCS ${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt
        DSTS ${FLUID_INFERENCE_INSTALL_DIR})

copy_part_of_thrid_party(inference_lib_dist ${FLUID_INFERENCE_INSTALL_DIR})

set(src_dir "${PADDLE_SOURCE_DIR}/paddle/fluid")
if(WIN32)
    set(paddle_fluid_lib ${PADDLE_BINARY_DIR}/paddle/fluid/inference/${CMAKE_BUILD_TYPE}/*paddle_fluid.*)
else(WIN32)
    set(paddle_fluid_lib ${PADDLE_BINARY_DIR}/paddle/fluid/inference/libpaddle_fluid.*)
endif(WIN32)

copy(inference_lib_dist
        SRCS  ${src_dir}/inference/api/paddle_*.h ${paddle_fluid_lib}
        DSTS  ${FLUID_INFERENCE_INSTALL_DIR}/paddle/include ${FLUID_INFERENCE_INSTALL_DIR}/paddle/lib)

copy(inference_lib_dist
        SRCS  ${CMAKE_BINARY_DIR}/paddle/fluid/framework/framework.pb.h
        DSTS  ${FLUID_INFERENCE_INSTALL_DIR}/paddle/include/internal)

# CAPI inference library for only inference
set(FLUID_INFERENCE_C_INSTALL_DIR "${CMAKE_BINARY_DIR}/fluid_inference_c_install_dir" CACHE STRING
"A path setting CAPI fluid inference shared")
copy_part_of_thrid_party(inference_lib_dist ${FLUID_INFERENCE_C_INSTALL_DIR})

set(src_dir "${PADDLE_SOURCE_DIR}/paddle/fluid")
if(WIN32)
    set(paddle_fluid_c_lib ${PADDLE_BINARY_DIR}/paddle/fluid/inference/capi/${CMAKE_BUILD_TYPE}/paddle_fluid_c.*)
else(WIN32)
    set(paddle_fluid_c_lib ${PADDLE_BINARY_DIR}/paddle/fluid/inference/capi/libpaddle_fluid_c.*)
endif(WIN32)

copy(inference_lib_dist
        SRCS  ${src_dir}/inference/capi/paddle_c_api.h  ${paddle_fluid_c_lib}
        DSTS  ${FLUID_INFERENCE_C_INSTALL_DIR}/paddle/include ${FLUID_INFERENCE_C_INSTALL_DIR}/paddle/lib)

# fluid library for both train and inference
set(fluid_lib_deps inference_lib_dist)
add_custom_target(fluid_lib_dist ALL DEPENDS ${fluid_lib_deps})

set(dst_dir "${FLUID_INSTALL_DIR}/paddle/fluid")
set(module "inference")
copy(fluid_lib_dist
        SRCS ${src_dir}/${module}/*.h ${src_dir}/${module}/api/paddle_*.h ${paddle_fluid_lib}
        DSTS ${dst_dir}/${module} ${dst_dir}/${module} ${dst_dir}/${module}
        )

set(module "framework")
set(framework_lib_deps framework_proto)
add_dependencies(fluid_lib_dist ${framework_lib_deps})
copy(fluid_lib_dist
        SRCS ${src_dir}/${module}/*.h ${src_dir}/${module}/details/*.h ${PADDLE_BINARY_DIR}/paddle/fluid/framework/trainer_desc.pb.h ${PADDLE_BINARY_DIR}/paddle/fluid/framework/framework.pb.h ${PADDLE_BINARY_DIR}/paddle/fluid/framework/data_feed.pb.h ${src_dir}/${module}/ir/memory_optimize_pass/*.h
        ${src_dir}/${module}/ir/*.h ${src_dir}/${module}/fleet/*.h
        DSTS ${dst_dir}/${module} ${dst_dir}/${module}/details ${dst_dir}/${module} ${dst_dir}/${module} ${dst_dir}/${module} ${dst_dir}/${module}/ir/memory_optimize_pass ${dst_dir}/${module}/ir ${dst_dir}/${module}/fleet)

set(module "operators")
copy(fluid_lib_dist
        SRCS ${src_dir}/${module}/reader/blocking_queue.h
        DSTS ${dst_dir}/${module}/reader/
        )

set(module "memory")
copy(fluid_lib_dist
        SRCS ${src_dir}/${module}/*.h ${src_dir}/${module}/detail/*.h ${src_dir}/${module}/allocation/*.h
        DSTS ${dst_dir}/${module} ${dst_dir}/${module}/detail ${dst_dir}/${module}/allocation
        )

set(module "platform")
set(platform_lib_deps profiler_proto)
add_dependencies(fluid_lib_dist ${platform_lib_deps})
copy(fluid_lib_dist
        SRCS ${src_dir}/${module}/*.h ${src_dir}/${module}/dynload/*.h ${src_dir}/${module}/details/*.h ${PADDLE_BINARY_DIR}/paddle/fluid/platform/profiler.pb.h ${PADDLE_BINARY_DIR}/paddle/fluid/platform/error_codes.pb.h
        DSTS ${dst_dir}/${module} ${dst_dir}/${module}/dynload ${dst_dir}/${module}/details ${dst_dir}/${module} ${dst_dir}/${module}
        )

set(module "string")
copy(fluid_lib_dist
        SRCS ${src_dir}/${module}/*.h ${src_dir}/${module}/tinyformat/*.h
        DSTS ${dst_dir}/${module} ${dst_dir}/${module}/tinyformat
        )

set(module "imperative")
copy(fluid_lib_dist
        SRCS ${src_dir}/${module}/*.h ${src_dir}/${module}/jit/*.h 
        DSTS ${dst_dir}/${module} ${dst_dir}/${module}/jit
        )

set(module "pybind")
copy(fluid_lib_dist
        SRCS ${CMAKE_CURRENT_BINARY_DIR}/paddle/fluid/${module}/pybind.h
        DSTS ${dst_dir}/${module}
        )

set(dst_dir "${FLUID_INSTALL_DIR}/third_party/eigen3")
copy(inference_lib_dist
        SRCS ${EIGEN_INCLUDE_DIR}/Eigen/Core ${EIGEN_INCLUDE_DIR}/Eigen/src ${EIGEN_INCLUDE_DIR}/unsupported/Eigen
        DSTS ${dst_dir}/Eigen ${dst_dir}/Eigen ${dst_dir}/unsupported)

set(dst_dir "${FLUID_INSTALL_DIR}/third_party/boost")
copy(inference_lib_dist
        SRCS ${BOOST_INCLUDE_DIR}/boost
        DSTS ${dst_dir})

set(dst_dir "${FLUID_INSTALL_DIR}/third_party/dlpack")
copy(inference_lib_dist
        SRCS ${DLPACK_INCLUDE_DIR}/dlpack
        DSTS ${dst_dir})

set(dst_dir "${FLUID_INSTALL_DIR}/third_party/install/zlib")
copy(inference_lib_dist
        SRCS ${ZLIB_INCLUDE_DIR} ${ZLIB_LIBRARIES}
        DSTS ${dst_dir} ${dst_dir}/lib)

# CMakeCache Info
copy(fluid_lib_dist
        SRCS ${FLUID_INFERENCE_INSTALL_DIR}/third_party ${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt
        DSTS ${FLUID_INSTALL_DIR} ${FLUID_INSTALL_DIR}
        )

# paddle fluid version
function(version version_file)
    execute_process(
            COMMAND ${GIT_EXECUTABLE} log --pretty=format:%H -1
            WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}
            OUTPUT_VARIABLE PADDLE_GIT_COMMIT)
    file(WRITE ${version_file}
            "GIT COMMIT ID: ${PADDLE_GIT_COMMIT}\n"
            "WITH_MKL: ${WITH_MKL}\n"
            "WITH_MKLDNN: ${WITH_MKLDNN}\n"
            "WITH_GPU: ${WITH_GPU}\n")
    if(WITH_GPU)
        file(APPEND ${version_file}
                "CUDA version: ${CUDA_VERSION}\n"
                "CUDNN version: v${CUDNN_MAJOR_VERSION}\n")
    endif()
    if(TENSORRT_FOUND)
        file(APPEND ${version_file}
                "WITH_TENSORRT: ${TENSORRT_FOUND}\n")
    endif()
    
endfunction()
version(${FLUID_INSTALL_DIR}/version.txt)
version(${FLUID_INFERENCE_INSTALL_DIR}/version.txt)
version(${FLUID_INFERENCE_C_INSTALL_DIR}/version.txt)
