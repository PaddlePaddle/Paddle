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

add_library(ngraph INTERFACE)

IF(WIN32 OR APPLE)
    MESSAGE(WARNING
        "Windows or Mac is not supported with nGraph in Paddle yet."
        "Force WITH_NGRAPH=OFF")
    SET(WITH_NGRAPH OFF CACHE STRING "Disable nGraph in Windows and MacOS" FORCE)
ENDIF()

IF(${WITH_NGRAPH} AND NOT ${WITH_MKLDNN})
    MESSAGE(WARNING
        "nGraph needs mkl-dnn to be enabled."
        "Force WITH_NGRAPH=OFF")
    SET(WITH_NGRAPH OFF CACHE STRING "Disable nGraph if mkl-dnn is disabled" FORCE)
ENDIF()

IF(NOT ${WITH_NGRAPH})
    return()
ENDIF()

INCLUDE(GNUInstallDirs)

INCLUDE(ExternalProject)

SET(NGRAPH_PROJECT         "extern_ngraph")
SET(NGRAPH_GIT_TAG         "127e0dedfaac8c6f2b148cc03bf5f67ac5fbe6fe")
SET(NGRAPH_SOURCES_DIR     ${THIRD_PARTY_PATH}/ngraph)
SET(NGRAPH_INSTALL_DIR     ${THIRD_PARTY_PATH}/install/ngraph)
SET(NGRAPH_INC_DIR         ${NGRAPH_INSTALL_DIR}/include)
SET(NGRAPH_LIB_DIR         ${NGRAPH_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR})
SET(NGRAPH_SHARED_LIB_NAME libngraph.so)
SET(NGRAPH_CPU_LIB_NAME    libcpu_backend.so)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    SET(NGRAPH_TBB_LIB_NAME    libtbb_debug.so.2)
else()
    SET(NGRAPH_TBB_LIB_NAME    libtbb.so.2)
endif()
SET(NGRAPH_GIT_REPO        "https://github.com/NervanaSystems/ngraph.git")
SET(NGRAPH_SHARED_LIB      ${NGRAPH_LIB_DIR}/${NGRAPH_SHARED_LIB_NAME})
SET(NGRAPH_CPU_LIB         ${NGRAPH_LIB_DIR}/${NGRAPH_CPU_LIB_NAME})
SET(NGRAPH_TBB_LIB         ${NGRAPH_LIB_DIR}/${NGRAPH_TBB_LIB_NAME})

ExternalProject_Add(
    ${NGRAPH_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    DEPENDS                  ${MKLDNN_PROJECT} ${MKLML_PROJECT}
    GIT_REPOSITORY           ${NGRAPH_GIT_REPO}
    GIT_TAG                  ${NGRAPH_GIT_TAG}
    PREFIX                   ${NGRAPH_SOURCES_DIR}
    UPDATE_COMMAND           ""
    CMAKE_GENERATOR          ${CMAKE_GENERATOR}
    CMAKE_GENERATOR_PLATFORM ${CMAKE_GENERATOR_PLATFORM}
    CMAKE_GENERATOR_TOOLSET  ${CMAKE_GENERATOR_TOOLSET}
    CMAKE_ARGS               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    CMAKE_ARGS               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    CMAKE_ARGS               -DCMAKE_INSTALL_PREFIX=${NGRAPH_INSTALL_DIR}
    CMAKE_ARGS               -DNGRAPH_UNIT_TEST_ENABLE=FALSE
    CMAKE_ARGS               -DNGRAPH_TOOLS_ENABLE=FALSE
    CMAKE_ARGS               -DNGRAPH_INTERPRETER_ENABLE=FALSE
    CMAKE_ARGS               -DNGRAPH_DEX_ONLY=TRUE
    CMAKE_ARGS               -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    CMAKE_ARGS               -DMKLDNN_INCLUDE_DIR=${MKLDNN_INC_DIR}
    CMAKE_ARGS               -DMKLDNN_LIB_DIR=${MKLDNN_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}
    CMAKE_ARGS               -DMKLML_LIB_DIR=${MKLML_INSTALL_DIR}/lib
)

add_dependencies(ngraph ${NGRAPH_PROJECT})
target_compile_definitions(ngraph INTERFACE -DPADDLE_WITH_NGRAPH)
target_include_directories(ngraph INTERFACE ${NGRAPH_INC_DIR})
target_link_libraries(ngraph INTERFACE ${NGRAPH_SHARED_LIB})
