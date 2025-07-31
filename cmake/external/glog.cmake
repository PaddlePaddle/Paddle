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

include(ExternalProject)

add_definitions(-DGLOG_NO_ABBREVIATED_SEVERITIES)

set(GLOG_PREFIX_DIR ${THIRD_PARTY_PATH}/glog)
set(GLOG_INSTALL_DIR ${THIRD_PARTY_PATH}/install/glog)
set(GLOG_INCLUDE_DIR
    "${GLOG_INSTALL_DIR}/include"
    CACHE PATH "glog include directory." FORCE)
set(GLOG_TAG v0.4.0)
set(SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/glog)
if(WIN32)
  set(GLOG_LIBRARIES
      "${GLOG_INSTALL_DIR}/lib/glog.lib"
      CACHE FILEPATH "glog library." FORCE)
  set(GLOG_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4267 /wd4530")
  add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
else()
  set(GLOG_LIBRARIES
      "${GLOG_INSTALL_DIR}/lib/libglog.a"
      CACHE FILEPATH "glog library." FORCE)
  set(GLOG_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif()

include_directories(${GLOG_INCLUDE_DIR})

ExternalProject_Add(
  extern_glog
  ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
  SOURCE_DIR ${SOURCE_DIR}
  DEPENDS gflags
  PREFIX ${GLOG_PREFIX_DIR}
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_FLAGS=${GLOG_CMAKE_CXX_FLAGS}
             -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
             -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
             -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
             -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
             -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
             -DCMAKE_INSTALL_PREFIX=${GLOG_INSTALL_DIR}
             -DCMAKE_INSTALL_LIBDIR=${GLOG_INSTALL_DIR}/lib
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DWITH_GFLAGS=OFF
             -DBUILD_TESTING=OFF
             -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
             ${EXTERNAL_OPTIONAL_ARGS}
  CMAKE_CACHE_ARGS
    -DCMAKE_INSTALL_PREFIX:PATH=${GLOG_INSTALL_DIR}
    -DCMAKE_INSTALL_LIBDIR:PATH=${GLOG_INSTALL_DIR}/lib
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
  BUILD_BYPRODUCTS ${GLOG_LIBRARIES})

add_library(glog STATIC IMPORTED GLOBAL)
set_property(TARGET glog PROPERTY IMPORTED_LOCATION ${GLOG_LIBRARIES})
add_dependencies(glog extern_glog gflags)
link_libraries(glog)
