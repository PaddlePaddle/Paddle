# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
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

set(VARIANT_SOURCES_DIR ${THIRD_PARTY_PATH}/variant)

set(VARIANT_INCLUDE_DIR "${THIRD_PARTY_PATH}/install/variant/include"
    CACHE PATH "variant include directory." FORCE)

include_directories(${VARIANT_INCLUDE_DIR})

ExternalProject_Add(
    variant
    ${EXTERNAL_PROJECT_LOG_ARGS}
    GIT_REPOSITORY    https://github.com/mapbox/variant
    PREFIX            ${VARIANT_SOURCES_DIR}
    UPDATE_COMMAND    ""
    CONFIGURE_COMMAND ""
    PATCH_COMMAND     ""
    UPDATE_COMMAND    ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
)

ExternalProject_Add_Step(variant variant_install
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          <SOURCE_DIR>/include
          ${VARIANT_INCLUDE_DIR}
  DEPENDEES install
)

list(APPEND external_project_dependencies variant)
