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

#FIXME:(gongwb) Move brpc's gtest dependency.

if(WITH_TESTING)
  enable_testing()
endif()

include(GNUInstallDirs)
include(ExternalProject)

set(GTEST_PREFIX_DIR ${THIRD_PARTY_PATH}/gtest)
set(GTEST_INSTALL_DIR ${THIRD_PARTY_PATH}/install/gtest)
set(GTEST_INCLUDE_DIR
    "${GTEST_INSTALL_DIR}/include"
    CACHE PATH "gtest include directory." FORCE)
set(GTEST_REPOSITORY ${GIT_URL}/google/googletest.git)
set(GTEST_TAG release-1.8.1)

include_directories(${GTEST_INCLUDE_DIR})

if(WIN32)
  set(GTEST_LIBRARIES
      "${GTEST_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/gtest.lib"
      CACHE FILEPATH "gtest libraries." FORCE)
  set(GTEST_MAIN_LIBRARIES
      "${GTEST_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/gtest_main.lib"
      CACHE FILEPATH "gtest main libraries." FORCE)
  set(GMOCK_LIBRARIES
      "${GTEST_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/libgmock.lib"
      CACHE FILEPATH "gmock libraries." FORCE)
  string(REPLACE "/w " "" GTEST_CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  string(REPLACE "/w " "" GTEST_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "/W0 " "" GTEST_CMAKE_C_FLAGS "${GTEST_CMAKE_C_FLAGS}")
  string(REPLACE "/W0 " "" GTEST_CMAKE_CXX_FLAGS "${GTEST_CMAKE_CXX_FLAGS}")
else()
  set(GTEST_LIBRARIES
      "${GTEST_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/libgtest.a"
      CACHE FILEPATH "gtest libraries." FORCE)
  set(GTEST_MAIN_LIBRARIES
      "${GTEST_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/libgtest_main.a"
      CACHE FILEPATH "gtest main libraries." FORCE)
  set(GMOCK_LIBRARIES
      "${GTEST_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/libgmock.a"
      CACHE FILEPATH "gmock libraries." FORCE)
  set(GTEST_CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  set(GTEST_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(WITH_MKLML)
  # wait for mklml downloading completed
  set(GTEST_DEPENDS ${MKLML_PROJECT})
endif()

ExternalProject_Add(
  extern_gtest
  ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
  GIT_REPOSITORY ${GTEST_REPOSITORY}
  GIT_TAG ${GTEST_TAG}
  DEPENDS ${GTEST_DEPENDS}
  PREFIX ${GTEST_PREFIX_DIR}
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_FLAGS=${GTEST_CMAKE_CXX_FLAGS}
             -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
             -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
             -DCMAKE_C_FLAGS=${GTEST_CMAKE_C_FLAGS}
             -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
             -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
             -DCMAKE_INSTALL_PREFIX=${GTEST_INSTALL_DIR}
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DBUILD_GMOCK=ON
             -Dgtest_disable_pthreads=ON
             -Dgtest_force_shared_crt=ON
             -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
             ${EXTERNAL_OPTIONAL_ARGS}
  CMAKE_CACHE_ARGS
    -DCMAKE_INSTALL_PREFIX:PATH=${GTEST_INSTALL_DIR}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
  BUILD_BYPRODUCTS ${GTEST_LIBRARIES}
  BUILD_BYPRODUCTS ${GTEST_MAIN_LIBRARIES}
  BUILD_BYPRODUCTS ${GMOCK_LIBRARIES})

add_library(gtest STATIC IMPORTED GLOBAL)
set_property(TARGET gtest PROPERTY IMPORTED_LOCATION ${GTEST_LIBRARIES})
add_dependencies(gtest extern_gtest)

add_library(gtest_main STATIC IMPORTED GLOBAL)
set_property(TARGET gtest_main PROPERTY IMPORTED_LOCATION
                                        ${GTEST_MAIN_LIBRARIES})
add_dependencies(gtest_main extern_gtest)

add_library(gmock STATIC IMPORTED GLOBAL)
set_property(TARGET gmock PROPERTY IMPORTED_LOCATION ${GMOCK_LIBRARIES})
add_dependencies(gmock extern_gtest)
