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

if(WITH_IPU)
  set(POPLAR_DIR CACHE PATH "Path to a Poplar install")
  set(POPART_DIR CACHE PATH "Path to a Popart install")
  set(POPLAR_SDK_DIR CACHE PATH "Path to an extracted SDK archive or to a Poplar & Popart install directory (Will populate POPLAR_DIR and POPART_DIR)")

  # support setting SDK both from environment variable or command line arguments

  if(DEFINED ENV{POPLAR_SDK_DIR})
    set(POPLAR_SDK_DIR $ENV{POPLAR_SDK_DIR})
  endif()
  if(EXISTS ${POPLAR_SDK_DIR})
    execute_process(COMMAND find ${POPLAR_SDK_DIR}/ -maxdepth 1 -type d -name "popart*"
      OUTPUT_VARIABLE POPART_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND find ${POPLAR_SDK_DIR}/ -maxdepth 1 -type d -name "poplar-*" -o -name "poplar"
      OUTPUT_VARIABLE POPLAR_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
  if(DEFINED ENV{POPLAR_DIR})
    set(POPLAR_DIR $ENV{POPLAR_DIR})
  endif()
  if(DEFINED ENV{POPART_DIR})
    set(POPART_DIR $ENV{POPART_DIR})
  endif()

  if(EXISTS ${POPLAR_DIR})
    message("POPLAR_DIR is ${POPLAR_DIR}")
    if(NOT IS_DIRECTORY "${POPLAR_DIR}")
      message(FATAL_ERROR "Couldn't find a \"poplar\" or \"poplar-*\" folder in '${POPLAR_SDK_DIR}'")
    endif()
    list(APPEND CMAKE_PREFIX_PATH ${POPLAR_DIR})
    set(ENABLE_POPLAR_CMD "source ${POPLAR_DIR}/enable.sh")
    find_package(poplar REQUIRED)
    include_directories("${POPLAR_DIR}/include")
    link_directories("${POPLAR_DIR}/lib")
  endif()
  if(NOT poplar_FOUND)
      message(FATAL_ERROR "You must provide a path to a Poplar install using -DPOPLAR_DIR=/path/to/popart/build/install")
  endif()
  if(EXISTS ${POPART_DIR})
    message("POPART_DIR is ${POPART_DIR}")
    if(NOT IS_DIRECTORY "${POPART_DIR}")
      message(FATAL_ERROR "Couldn't find a \"popart*\" folder in '${POPLAR_SDK_DIR}'")
    endif()
    list(APPEND CMAKE_PREFIX_PATH ${POPART_DIR})
    set(ENABLE_POPART_CMD "source ${POPART_DIR}/enable.sh")
    find_package(popart REQUIRED COMPONENTS popart-only)
    include_directories("${POPART_DIR}/include")
    link_directories("${POPART_DIR}/lib")
  endif()
  if(NOT popart_FOUND)
    message(FATAL_ERROR "You must provide a path to a Popart build using -DPOPART_DIR=/path/to/popart/build")
  endif()

  add_definitions(-DONNX_NAMESPACE=onnx)
  add_custom_target(extern_poplar DEPENDS poplar popart-only)
endif()
