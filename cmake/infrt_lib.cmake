# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

function(copy TARGET)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SRCS DSTS)
  cmake_parse_arguments(copy_lib "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  list(LENGTH copy_lib_SRCS copy_lib_SRCS_len)
  list(LENGTH copy_lib_DSTS copy_lib_DSTS_len)
  if(NOT ${copy_lib_SRCS_len} EQUAL ${copy_lib_DSTS_len})
    message(
      FATAL_ERROR
        "${TARGET} source numbers are not equal to destination numbers")
  endif()
  math(EXPR len "${copy_lib_SRCS_len} - 1")
  foreach(index RANGE ${len})
    list(GET copy_lib_SRCS ${index} src)
    list(GET copy_lib_DSTS ${index} dst)
    add_custom_command(
      TARGET ${TARGET}
      POST_BUILD
      COMMAND mkdir -p "${dst}"
      COMMAND cp -r "${src}" "${dst}"
      COMMENT "copying ${src} -> ${dst}")
  endforeach()
endfunction()

function(copy_part_of_thrid_party TARGET DST)
  set(dst_dir "${DST}/third_party/install/glog")
  copy(
    ${TARGET}
    SRCS ${GLOG_INCLUDE_DIR} ${GLOG_LIBRARIES}
    DSTS ${dst_dir} ${dst_dir}/lib)
endfunction()

# paddle fluid version
function(version version_file)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} log --pretty=format:%H -1
    WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}
    OUTPUT_VARIABLE PADDLE_GIT_COMMIT)
  file(WRITE ${version_file} "GIT COMMIT ID: ${PADDLE_GIT_COMMIT}\n")
  file(APPEND ${version_file}
       "CXX compiler version: ${CMAKE_CXX_COMPILER_VERSION}\n")
endfunction()
