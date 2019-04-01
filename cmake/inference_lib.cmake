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
function(copy TARGET)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DSTS DEPS)
    cmake_parse_arguments(copy_lib "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set(fluid_lib_dist_dep ${TARGET} ${fluid_lib_dist_dep} PARENT_SCOPE)

    list(LENGTH copy_lib_SRCS copy_lib_SRCS_len)
    list(LENGTH copy_lib_DSTS copy_lib_DSTS_len)
    if (NOT ${copy_lib_SRCS_len} EQUAL ${copy_lib_DSTS_len})
        message(FATAL_ERROR "${TARGET} source numbers are not equal to destination numbers")
    endif ()
    math(EXPR len "${copy_lib_SRCS_len} - 1")

    add_custom_target(${TARGET} DEPENDS ${copy_lib_DEPS})
    foreach (index RANGE ${len})
        list(GET copy_lib_SRCS ${index} src)
        list(GET copy_lib_DSTS ${index} dst)
        if (WIN32)
            if(IS_DIRECTORY ${src})
                get_filename_component(last_path ${src} NAME)
                string(APPEND dst "/" ${last_path})
                add_custom_command(TARGET ${TARGET} PRE_BUILD
                        COMMAND ${CMAKE_COMMAND} -E make_directory "${dst}"
                        )
                if(EXISTS ${src})
                    add_custom_command(TARGET ${TARGET} PRE_BUILD
                            COMMAND cmake -E copy_directory "${src}" "${dst}"
                            COMMENT "copying ${src} -> ${dst}")
                else()
                    message(WARNING "${src} not exist!")
                endif()
            else()
                # windows cmd shell will not expand wildcard automatically.
                # below expand the files, and copy them by rules.
                file(GLOB src_files ${src})
                if (NOT "${src_files}" STREQUAL "")
                    list(REMOVE_DUPLICATES src_files)
                endif ()
                add_custom_command(TARGET ${TARGET} PRE_BUILD
                        COMMAND ${CMAKE_COMMAND} -E make_directory "${dst}"
                        )
                foreach (src_file ${src_files})
                    add_custom_command(TARGET ${TARGET} PRE_BUILD
                            COMMAND ${CMAKE_COMMAND} -E copy "${src_file}" "${dst}"
                            COMMENT "copying ${src_file} -> ${dst}")
                endforeach ()
            endif()
        else (WIN32) # not windows
            add_custom_command(TARGET ${TARGET} PRE_BUILD
                    COMMAND mkdir -p "${dst}"
                    COMMAND cp -r "${src}" "${dst}"
                    COMMENT "copying ${src} -> ${dst}")
        endif (WIN32) # not windows
    endforeach ()
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

set(dst_dir "${FLUID_INSTALL_DIR}/third_party/install/xxhash")
copy(xxhash_lib
        SRCS ${XXHASH_INCLUDE_DIR} ${XXHASH_LIBRARIES}
        DSTS ${dst_dir} ${dst_dir}/lib
        DEPS xxhash
        )

if (NOT PROTOBUF_FOUND OR WIN32)
    set(dst_dir "${FLUID_INSTALL_DIR}/third_party/install/protobuf")
    copy(protobuf_lib
            SRCS ${PROTOBUF_INCLUDE_DIR} ${PROTOBUF_LIBRARY}
            DSTS ${dst_dir} ${dst_dir}/lib
            DEPS extern_protobuf
            )
endif ()

if (WITH_MKLML)
    set(dst_dir "${FLUID_INSTALL_DIR}/third_party/install/mklml")
    copy(mklml_lib
            SRCS ${MKLML_LIB} ${MKLML_IOMP_LIB} ${MKLML_INC_DIR}
            DSTS ${dst_dir}/lib ${dst_dir}/lib ${dst_dir}
            DEPS mklml
            )
elseif (NOT CBLAS_FOUND OR WIN32)
    set(dst_dir "${FLUID_INSTALL_DIR}/third_party/install/openblas")
    copy(openblas_lib
            SRCS ${CBLAS_INSTALL_DIR}/lib ${CBLAS_INSTALL_DIR}/include
            DSTS ${dst_dir} ${dst_dir}
            DEPS extern_openblas
            )
endif ()

if (WITH_GPU AND NOT WIN32)
    set(dgc_dir "${FLUID_INSTALL_DIR}/third_party/install/dgc")
    copy(dgc_lib
            SRCS ${DGC_INSTALL_DIR}/lib ${DGC_INSTALL_DIR}/include
            DSTS ${dgc_dir} ${dgc_dir}
            DEPS dgc)
endif()


if (WITH_MKLDNN)
    set(dst_dir "${FLUID_INSTALL_DIR}/third_party/install/mkldnn")
    copy(mkldnn_lib
            SRCS ${MKLDNN_INC_DIR} ${MKLDNN_SHARED_LIB}
            DSTS ${dst_dir} ${dst_dir}/lib
            DEPS mkldnn_shared_lib
            )
endif ()

if (WITH_NGRAPH)
    set(dst_dir "${FLUID_INSTALL_DIR}/third_party/install/ngraph")
    copy(ngraph_lib
            SRCS ${NGRAPH_INC_DIR} ${NGRAPH_LIB_DIR}
            DSTS ${dst_dir} ${dst_dir}
            DEPS ngraph
            )
endif ()

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

if (WITH_WBAES)
    set(dst_dir "${FLUID_INSTALL_DIR}/third_party/install/wbaes")
    copy(wbaes_lib
            SRCS ${WBAES_INC_DIR} ${WBAES_LIB}
            DSTS ${dst_dir} ${dst_dir}/lib
            DEPS wbaes)
endif ()

set(dst_dir "${FLUID_INSTALL_DIR}/third_party/install/zlib")
copy(zlib_lib
        SRCS ${ZLIB_INCLUDE_DIR} ${ZLIB_LIBRARIES}
        DSTS ${dst_dir} ${dst_dir}/lib
        DEPS zlib)

# paddle fluid module
set(src_dir "${PADDLE_SOURCE_DIR}/paddle/fluid")
set(dst_dir "${FLUID_INSTALL_DIR}/paddle/fluid")
set(module "framework")
if (NOT WIN32)
    set(framework_lib_deps framework_py_proto)
endif (NOT WIN32)
copy(framework_lib DEPS ${framework_lib_deps}
        SRCS ${src_dir}/${module}/*.h ${src_dir}/${module}/details/*.h ${PADDLE_BINARY_DIR}/paddle/fluid/framework/framework.pb.h
        ${src_dir}/${module}/ir/*.h
        DSTS ${dst_dir}/${module} ${dst_dir}/${module}/details ${dst_dir}/${module} ${dst_dir}/${module}/ir
        )

set(module "memory")
copy(memory_lib
        SRCS ${src_dir}/${module}/*.h ${src_dir}/${module}/detail/*.h ${src_dir}/${module}/allocation/*.h
        DSTS ${dst_dir}/${module} ${dst_dir}/${module}/detail ${dst_dir}/${module}/allocation
        )

set(inference_deps paddle_fluid_shared paddle_fluid)

set(module "inference/api")
if (WITH_ANAKIN AND WITH_MKL)
    copy(anakin_inference_lib DEPS paddle_inference_api inference_anakin_api
            SRCS
            ${PADDLE_BINARY_DIR}/paddle/fluid/inference/api/libinference_anakin_api* # compiled anakin api
            ${ANAKIN_INSTALL_DIR} # anakin release
            DSTS ${FLUID_INSTALL_DIR}/third_party/install/anakin ${FLUID_INSTALL_DIR}/third_party/install/anakin)
    list(APPEND inference_deps anakin_inference_lib)
endif ()

if (TENSORRT_FOUND)
    copy(tensorrt_lib DEPS ${inference_deps} 
        SRCS ${TENSORRT_ROOT}/include/Nv*.h ${TENSORRT_ROOT}/lib/libnvinfer*
        DSTS ${FLUID_INSTALL_DIR}/third_party/install/tensorrt/include ${FLUID_INSTALL_DIR}/third_party/install/tensorrt/lib)
endif () 


set(module "inference")
if(WIN32)
    set(paddle_fluid_lib ${PADDLE_BINARY_DIR}/paddle/fluid/inference/${CMAKE_BUILD_TYPE}/libpaddle_fluid.*)
else(WIN32)
    set(paddle_fluid_lib ${PADDLE_BINARY_DIR}/paddle/fluid/inference/libpaddle_fluid.*)
endif(WIN32)
copy(inference_lib DEPS ${inference_deps}
  SRCS ${src_dir}/${module}/*.h ${paddle_fluid_lib}
       ${src_dir}/${module}/api/paddle_*.h
  DSTS ${dst_dir}/${module} ${dst_dir}/${module} ${dst_dir}/${module}
        )

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

# This command generates a complete fluid library for both train and inference
add_custom_target(fluid_lib_dist DEPENDS ${fluid_lib_dist_dep})

# Following commands generate a inference-only fluid library
# third_party, version.txt and CMakeCache.txt are the same position with ${FLUID_INSTALL_DIR}
copy(third_party DEPS fluid_lib_dist
        SRCS ${FLUID_INSTALL_DIR}/third_party ${FLUID_INSTALL_DIR}/CMakeCache.txt
        DSTS ${FLUID_INFERENCE_INSTALL_DIR} ${FLUID_INFERENCE_INSTALL_DIR}
        )

# only need libpaddle_fluid.so/a and paddle_*.h for inference-only library
copy(inference_api_lib DEPS fluid_lib_dist
  SRCS ${paddle_fluid_lib}
       ${FLUID_INSTALL_DIR}/paddle/fluid/inference/paddle_*.h
  DSTS ${FLUID_INFERENCE_INSTALL_DIR}/paddle/lib ${FLUID_INFERENCE_INSTALL_DIR}/paddle/include
)

add_custom_target(inference_lib_dist DEPENDS third_party inference_api_lib)

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
    if (WITH_GPU)
        file(APPEND ${version_file}
                "CUDA version: ${CUDA_VERSION}\n"
                "CUDNN version: v${CUDNN_MAJOR_VERSION}\n")
    endif ()
endfunction()
version(${FLUID_INSTALL_DIR}/version.txt)
version(${FLUID_INFERENCE_INSTALL_DIR}/version.txt)
