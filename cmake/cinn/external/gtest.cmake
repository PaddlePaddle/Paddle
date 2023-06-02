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

if(WITH_TESTING)
  enable_testing()

  include(ExternalProject)

  set(GTEST_SOURCES_DIR ${CINN_THIRD_PARTY_PATH}/gtest)
  set(GTEST_INSTALL_DIR ${CINN_THIRD_PARTY_PATH}/install/gtest)
  set(GTEST_INCLUDE_DIR
      "${GTEST_INSTALL_DIR}/include"
      CACHE PATH "gtest include directory." FORCE)

  include_directories(${GTEST_INCLUDE_DIR})

  if(WIN32)
    set(GTEST_LIBRARIES
        "${GTEST_INSTALL_DIR}/lib/gtest.lib"
        CACHE FILEPATH "gtest libraries." FORCE)
    set(GTEST_MAIN_LIBRARIES
        "${GTEST_INSTALL_DIR}/lib/gtest_main.lib"
        CACHE FILEPATH "gtest main libraries." FORCE)
  else(WIN32)
    set(GTEST_LIBRARIES
        "${GTEST_INSTALL_DIR}/lib/libgtest.a"
        CACHE FILEPATH "gtest libraries." FORCE)
    set(GTEST_MAIN_LIBRARIES
        "${GTEST_INSTALL_DIR}/lib/libgtest_main.a"
        CACHE FILEPATH "gtest main libraries." FORCE)
  endif(WIN32)

  if(WITH_MKLML)
    # wait for mklml downloading completed
    set(GTEST_DEPENDS ${MKLML_PROJECT})
  endif()

  set(OPTIONAL_ARGS
      "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
      "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
      "-DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}"
      "-DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}"
      "-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}"
      "-DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}"
      "-DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}")

  if(ANDROID)
    set(OPTIONAL_ARGS
        ${OPTIONAL_ARGS}
        "-DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}"
        "-DCMAKE_SYSTEM_VERSION=${CMAKE_SYSTEM_VERSION}"
        "-DCMAKE_ANDROID_ARCH_ABI=${CMAKE_ANDROID_ARCH_ABI}"
        "-DCMAKE_ANDROID_NDK=${CMAKE_ANDROID_NDK}"
        "-DCMAKE_ANDROID_STL_TYPE=${CMAKE_ANDROID_STL_TYPE}")
  endif()

  ExternalProject_Add(
    extern_gtest
    ${EXTERNAL_PROJECT_LOG_ARGS}
    DEPENDS ${GTEST_DEPENDS}
    GIT_REPOSITORY "https://github.com/google/googletest.git"
    GIT_TAG "release-1.8.0"
    PREFIX ${GTEST_SOURCES_DIR}
    UPDATE_COMMAND ""
    CMAKE_ARGS ${OPTIONAL_ARGS}
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
    BUILD_BYPRODUCTS ${GTEST_LIBRARIES})

  add_library(gtest STATIC IMPORTED GLOBAL)
  set_property(TARGET gtest PROPERTY IMPORTED_LOCATION ${GTEST_LIBRARIES})
  add_dependencies(gtest extern_gtest)

  add_library(gtest_main STATIC IMPORTED GLOBAL)
  set_property(TARGET gtest_main PROPERTY IMPORTED_LOCATION
                                          ${GTEST_MAIN_LIBRARIES})
  add_dependencies(gtest_main extern_gtest)

endif()
