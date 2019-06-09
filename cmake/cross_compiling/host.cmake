# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

set(HOST_C_COMPILER $ENV{CC})
set(HOST_CXX_COMPILER $ENV{CXX})

if(NOT HOST_C_COMPILER)
    find_program(HOST_C_COMPILER NAMES gcc PATH
        /usr/bin
        /usr/local/bin)
endif()

if(NOT HOST_CXX_COMPILER)
    find_program(HOST_CXX_COMPILER NAMES g++ PATH
        /usr/bin
        /usr/local/bin)
endif()

if(NOT HOST_C_COMPILER OR NOT EXISTS ${HOST_C_COMPILER})
    MESSAGE(FATAL_ERROR "Cannot find host C compiler. export CC=/path/to/cc")
ENDIF()

if(NOT HOST_CXX_COMPILER OR NOT EXISTS ${HOST_CXX_COMPILER})
    MESSAGE(FATAL_ERROR "Cannot find host C compiler. export CC=/path/to/cc")
ENDIF()

MESSAGE(STATUS "Found host C compiler: " ${HOST_C_COMPILER})
MESSAGE(STATUS "Found host CXX compiler: " ${HOST_CXX_COMPILER})

