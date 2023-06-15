# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

set(CMAKE_SYSTEM_NAME QNX)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(QNX_HOST "$ENV{QNX_HOST}")
set(QNX_TARGET "$ENV{QNX_TARGET}")
if(NOT QNX_HOST)
  message(FATAL_ERROR "Environment variable QNX_HOST not set!")
endif()
if(NOT QNX_TARGET)
  message(FATAL_ERROR "Environment variable QNX_TARGET not set!")
endif()

set(QNX_PROCESSOR aarch64)
set(QNX_COMPILER_TARGET gcc_ntoaarch64le_cxx)
set(QNX_NTO_ARCH aarch64)

set(CMAKE_C_COMPILER ${QNX_HOST}/usr/bin/qcc)
set(CMAKE_C_COMPILER_TARGET ${QNX_COMPILER_TARGET})
set(CMAKE_CXX_COMPILER ${QNX_HOST}/usr/bin/q++)
set(CMAKE_CXX_COMPILER_TARGET ${QNX_COMPILER_TARGET})
# Don't use CMAKE_ASM_COMPILER_TARGET before version 3.14, refer to https://gitlab.kitware.com/cmake/cmake/-/merge_requests/3016
set(CMAKE_ASM_COMPILER ${QNX_HOST}/usr/bin/qcc -V${QNX_COMPILER_TARGET})
set(CMAKE_ASM_DEFINE_FLAG "-Wa,--defsym,")
set(CMAKE_STRIP
    ${QNX_HOST}/usr/bin/nto${QNX_NTO_ARCH}-strip
    CACHE PATH "QNX strip program" FORCE)
