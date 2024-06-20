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

if(WIN32)
  set(YAML_LIBRARIES
      "${YAML_INSTALL_DIR}/lib/yaml-cpp.lib"
      CACHE FILEPATH "yaml library." FORCE)
else()
  set(YAML_LIBRARIES
      "${YAML_INSTALL_DIR}/lib/libyaml-cpp.a"
      CACHE FILEPATH "yaml library." FORCE)
endif()

ExternalProject_Add(
  extern_yaml
  ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
  SOURCE_DIR ${SOURCE_DIR}
  PREFIX ${YAML_PREFIX_DIR}
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
             -DYAML_BUILD_SHARED_LIBS=OFF
             -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
             -DCMAKE_INSTALL_PREFIX=${YAML_INSTALL_DIR}
             -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
  CMAKE_CACHE_ARGS
    -DCMAKE_INSTALL_PREFIX:PATH=${YAML_INSTALL_DIR}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
  BUILD_BYPRODUCTS ${YAML_LIBRARIES})

add_library(yaml STATIC IMPORTED GLOBAL)
set_property(TARGET yaml PROPERTY IMPORTED_LOCATION ${YAML_LIBRARIES})
add_dependencies(yaml extern_yaml)
link_directories(${YAML_INSTALL_DIR}/lib/)
target_link_libraries(yaml INTERFACE ${YAML_LIBRARIES})
