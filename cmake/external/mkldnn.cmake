# Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.
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

IF(NOT ${WITH_MKLDNN})
  return()
ENDIF(NOT ${WITH_MKLDNN})

INCLUDE(ExternalProject)

SET(MKLDNN_PROJECT        "extern_mkldnn")
SET(MKLDNN_SOURCES_DIR    ${THIRD_PARTY_PATH}/mkldnn)
SET(MKLDNN_INSTALL_DIR    ${THIRD_PARTY_PATH}/install/mkldnn)
SET(MKLDNN_INC_DIR        "${MKLDNN_INSTALL_DIR}/include" CACHE PATH "mkldnn include directory." FORCE)

IF(WIN32 OR APPLE)
    MESSAGE(WARNING 
        "Windows or Mac is not supported with MKLDNN in Paddle yet."
        "Force WITH_MKLDNN=OFF")
    SET(WITH_MKLDNN OFF CACHE STRING "Disable MKLDNN in Windows and MacOS" FORCE)
    return()
ENDIF()

SET(MKLDNN_LIB "${MKLDNN_INSTALL_DIR}/lib/libmkldnn.so" CACHE FILEPATH "mkldnn library." FORCE)
MESSAGE(STATUS "Set ${MKLDNN_INSTALL_DIR}/lib to runtime path")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${MKLDNN_INSTALL_DIR}/lib")

INCLUDE_DIRECTORIES(${MKLDNN_INC_DIR}) # For MKLDNN code to include internal headers.
INCLUDE_DIRECTORIES(${THIRD_PARTY_PATH}/install) # For Paddle code to include mkldnn.h

IF(${CBLAS_PROVIDER} STREQUAL "MKLML")
    SET(MKLDNN_DEPENDS   ${MKLML_PROJECT})
    MESSAGE(STATUS "Build MKLDNN with MKLML ${MKLML_ROOT}")
ELSE()
    MESSAGE(FATAL_ERROR "Should enable MKLML when build MKLDNN")
ENDIF()
SET(MKLDNN_FLAG "-Wno-error=strict-overflow -Wno-error=unused-result")
SET(MKLDNN_FLAG "${MKLDNN_FLAG} -Wno-unused-result -Wno-unused-value")
SET(MKLDNN_CFLAG "${CMAKE_C_FLAGS} ${MKLDNN_FLAG}")
SET(MKLDNN_CXXFLAG "${CMAKE_CXX_FLAGS} ${MKLDNN_FLAG}")
ExternalProject_Add(
    ${MKLDNN_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    DEPENDS             ${MKLDNN_DEPENDS}
    GIT_REPOSITORY      "https://github.com/01org/mkl-dnn.git"
    GIT_TAG             "db3424ad44901513c03a1ea31ccaacdf633fbe9f"
    PREFIX              ${MKLDNN_SOURCES_DIR}
    UPDATE_COMMAND      ""
    CMAKE_ARGS          -DCMAKE_INSTALL_PREFIX=${MKLDNN_INSTALL_DIR}
    CMAKE_ARGS          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} 
    CMAKE_ARGS          -DMKLROOT=${MKLML_ROOT}
    CMAKE_ARGS          -DCMAKE_C_FLAGS=${MKLDNN_CFLAG}
    CMAKE_ARGS          -DCMAKE_CXX_FLAGS=${MKLDNN_CXXFLAG}
    CMAKE_ARGS          -DWITH_TEST=OFF -DWITH_EXAMPLE=OFF
    CMAKE_CACHE_ARGS    -DCMAKE_INSTALL_PREFIX:PATH=${MKLDNN_INSTALL_DIR}
                        -DMKLROOT:PATH=${MKLML_ROOT}
)

ADD_LIBRARY(shared_mkldnn SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET shared_mkldnn PROPERTY IMPORTED_LOCATION ${MKLDNN_LIB})
ADD_DEPENDENCIES(shared_mkldnn ${MKLDNN_PROJECT})
MESSAGE(STATUS "MKLDNN library: ${MKLDNN_LIB}")
add_definitions(-DPADDLE_WITH_MKLDNN)
LIST(APPEND external_project_dependencies shared_mkldnn)

# generate a static dummy target to track mkldnn dependencies
# for cc_library(xxx SRCS xxx.c DEPS mkldnn)
SET(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/mkldnn_dummy.c)
FILE(WRITE ${dummyfile} "const char * dummy = \"${dummyfile}\";")
ADD_LIBRARY(mkldnn STATIC ${dummyfile})
TARGET_LINK_LIBRARIES(mkldnn ${MKLDNN_LIB} ${MKLML_LIB} ${MKLML_IOMP_LIB})
ADD_DEPENDENCIES(mkldnn ${MKLDNN_PROJECT})

# copy the real so.0 lib to install dir
# it can be directly contained in wheel or capi
SET(MKLDNN_SHARED_LIB ${MKLDNN_INSTALL_DIR}/libmkldnn.so.0)
ADD_CUSTOM_COMMAND(OUTPUT ${MKLDNN_SHARED_LIB}
    COMMAND cp ${MKLDNN_LIB} ${MKLDNN_SHARED_LIB}
    DEPENDS mkldnn)
ADD_CUSTOM_TARGET(mkldnn_shared_lib ALL DEPENDS ${MKLDNN_SHARED_LIB})

IF(WITH_C_API)
  INSTALL(FILES ${MKLDNN_SHARED_LIB} DESTINATION lib)
ENDIF()

