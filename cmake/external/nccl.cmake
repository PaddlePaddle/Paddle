# Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.
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

IF(NOT ${WITH_GPU})
    MESSAGE(WARNING
        "WITH NCCL is not supported without GPU."
        "Force WITH_GPU=ON")
  return()
ENDIF(NOT ${WITH_GPU})

INCLUDE(ExternalProject)


SET(NCCL_SOURCES_DIR ${THIRD_PARTY_PATH}/nccl)
SET(NCCL_INSTALL_DIR ${THIRD_PARTY_PATH}/install/nccl)
SET(NCCL_INCLUDE_DIR "${NCCL_INSTALL_DIR}/include" CACHE PATH "nccl include directory." FORCE)

INCLUDE_DIRECTORIES(${NCCL_SOURCES_DIR}/src/extern_nccl/build/include)
INCLUDE_DIRECTORIES(${NCCL_INCLUDE_DIR})

IF(WIN32)
  SET(NCCL_LIBRARIES "${NCCL_INSTALL_DIR}/lib/libnccl_static.lib" CACHE FILEPATH "nccl library." FORCE)
ENDIF(WIN32)
  SET(NCCL_LIBRARIES "${NCCL_INSTALL_DIR}/lib/libnccl_static.a" CACHE FILEPATH "nccl library." FORCE)


# currently, nccl2 is not support in docker. So we use nccl1.
# the progress of nccl2 can be tracked in https://gitlab.com/nvidia/cuda/issues/10

ExternalProject_Add(
  extern_nccl
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX ${NCCL_SOURCES_DIR}
    DOWNLOAD_DIR ${NCCL_SOURCES_DIR}
    GIT_REPOSITORY  "https://github.com/NVIDIA/nccl.git"
    GIT_TAG         "v1.3.4-1"
    CONFIGURE_COMMAND ""
    CMAKE_COMMAND ""
    UPDATE_COMMAND  ""
    BUILD_IN_SOURCE 1
    BUILD_COMMAND    make -j 8
    INSTALL_COMMAND  make install
    INSTALL_DIR ${NCCL_INSTALL_DIR}
    TEST_COMMAND      ""
    )

MESSAGE(STATUS "nccl include: ${NCCL_INCLUDE_DIR}")
MESSAGE(STATUS "nccl source: ${NCCL_SOURCES_DIR}")

MESSAGE(STATUS "nccl library: ${NCCL_LIBRARIES}")

add_library(nccl INTERFACE)

add_dependencies(nccl extern_nccl)

LIST(APPEND external_project_dependencies nccl)
