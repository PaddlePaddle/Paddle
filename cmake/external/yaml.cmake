# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

set(YAML_PREFIX_DIR ${THIRD_PARTY_PATH}/yaml-cpp)
set(YAML_INSTALL_DIR ${THIRD_PARTY_PATH}/install/yaml-cpp)
set(YAML_INCLUDE_DIR ${YAML_INSTALL_DIR}/include)

set(SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/yaml-cpp)
set(SOURCE_INCLUDE_DIR ${SOURCE_DIR}/include)

include_directories(${YAML_INCLUDE_DIR})

set(YAML_BuildTests
    OFF
    CACHE INTERNAL "")

set(YAML_MSVC_SHARED_RT OFF)
if(WIN32)
  set(YAML_LIBRARIES
      "${YAML_INSTALL_DIR}/lib/yaml-cpp${CMAKE_STATIC_LIBRARY_SUFFIX}"
      CACHE FILEPATH "yaml library." FORCE)
  if(MSVC_STATIC_CRT)
    set(YAML_CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    set(YAML_CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    foreach(
      flag_var
      CMAKE_CXX_FLAGS
      CMAKE_CXX_FLAGS_DEBUG
      CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL
      CMAKE_CXX_FLAGS_RELWITHDEBINFO
      CMAKE_C_FLAGS
      CMAKE_C_FLAGS_DEBUG
      CMAKE_C_FLAGS_RELEASE
      CMAKE_C_FLAGS_MINSIZEREL
      CMAKE_C_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif()
    endforeach()
  else()
    set(YAML_MSVC_SHARED_RT ON)
    set(YAML_CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(YAML_CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    foreach(
      flag_var
      CMAKE_CXX_FLAGS
      CMAKE_CXX_FLAGS_DEBUG
      CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL
      CMAKE_CXX_FLAGS_RELWITHDEBINFO
      CMAKE_C_FLAGS
      CMAKE_C_FLAGS_DEBUG
      CMAKE_C_FLAGS_RELEASE
      CMAKE_C_FLAGS_MINSIZEREL
      CMAKE_C_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MT")
        string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
      endif()
    endforeach()
  endif()
else()
  set(YAML_LIBRARIES
      "${YAML_INSTALL_DIR}/lib/libyaml-cpp${CMAKE_STATIC_LIBRARY_SUFFIX}"
      CACHE FILEPATH "yaml library." FORCE)
  set(YAML_CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  set(YAML_CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
endif()

ExternalProject_Add(
  extern_yaml
  ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
  SOURCE_DIR ${SOURCE_DIR}
  PREFIX ${YAML_PREFIX_DIR}
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
             -DYAML_BUILD_SHARED_LIBS=OFF
             -DYAML_MSVC_SHARED_RT=${YAML_MSVC_SHARED_RT}
             -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
             -DCMAKE_INSTALL_PREFIX=${YAML_INSTALL_DIR}
             -DCMAKE_INSTALL_LIBDIR=${YAML_INSTALL_DIR}/lib
             -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
             -DCMAKE_CXX_FLAGS_RELEASE=${YAML_CMAKE_CXX_FLAGS_RELEASE}
             -DCMAKE_CXX_FLAGS_DEBUG=${YAML_CMAKE_CXX_FLAGS_DEBUG}
             -DWITH_ROCM=${WITH_ROCM}
  CMAKE_CACHE_ARGS
    -DCMAKE_INSTALL_PREFIX:PATH=${YAML_INSTALL_DIR}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
    ${EXTERNAL_OPTIONAL_ARGS}
  BUILD_BYPRODUCTS ${YAML_LIBRARIES})

add_library(yaml STATIC IMPORTED GLOBAL)
set_property(TARGET yaml PROPERTY IMPORTED_LOCATION ${YAML_LIBRARIES})
add_dependencies(yaml extern_yaml)
link_libraries(yaml)
