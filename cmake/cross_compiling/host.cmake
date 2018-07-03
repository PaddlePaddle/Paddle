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

# find host C compiler
IF(HOST_C_COMPILER)
    SET(HOST_C_COMPILER_NAME ${HOST_C_COMPILER})
ELSEIF(NOT $ENV{CC} STREQUAL "")
    SET(HOST_C_COMPILER_NAME $ENV{CC})
ELSE()
    SET(HOST_C_COMPILER_NAME cc)
ENDIF()

GET_FILENAME_COMPONENT(HOST_C_COMPILER_PATH ${HOST_C_COMPILER_NAME} PROGRAM)
IF(NOT HOST_C_COMPILER_PATH OR NOT EXISTS ${HOST_C_COMPILER_PATH})
    MESSAGE(FATAL_ERROR "Cannot find host C compiler, set host C compiler:\n"
            "\tcmake .. -DHOST_C_COMPILER=...")
ENDIF()

# find host CXX compiler
IF(HOST_CXX_COMPILER)
    SET(HOST_CXX_COMPILER_NAME ${HOST_CXX_COMPILER})
ELSEIF(NOT $ENV{CXX} STREQUAL "")
    SET(HOST_CXX_COMPILER_NAME $ENV{CXX})
ELSE()
    SET(HOST_CXX_COMPILER_NAME c++)
ENDIF()

GET_FILENAME_COMPONENT(HOST_CXX_COMPILER_PATH ${HOST_CXX_COMPILER_NAME} PROGRAM)
IF(NOT HOST_CXX_COMPILER_PATH OR NOT EXISTS ${HOST_CXX_COMPILER_PATH})
    MESSAGE(FATAL_ERROR "Cannot find host CXX compiler, set host CXX compiler:\n"
            "\tcmake .. -DHOST_CXX_COMPILER=...")
ENDIF()

SET(HOST_C_COMPILER ${HOST_C_COMPILER_PATH} CACHE PATH "Host C compiler")
SET(HOST_CXX_COMPILER ${HOST_CXX_COMPILER_PATH} CACHE PATH "Host CXX compiler")

MESSAGE(STATUS "Found host C compiler: " ${HOST_C_COMPILER})
MESSAGE(STATUS "Found host CXX compiler: " ${HOST_CXX_COMPILER})
