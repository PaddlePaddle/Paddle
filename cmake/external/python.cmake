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

INCLUDE(ExternalProject)

SET(PYTHON_SOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/python)
SET(PYTHON_INSTALL_DIR ${PROJECT_BINARY_DIR}/python)

if(MSVC)
  list(APPEND EXTERNAL_PROJECT_OPTIONAL_ARGS
    PATCH_COMMAND ${CMAKE_COMMAND}
      -DPYTHON_SRC_DIR:PATH=${_python_SOURCE_DIR}
      -P ${CMAKE_CURRENT_LIST_DIR}/PythonPatch.cmake
    )
endif()

if(APPLE)
  list(APPEND EXTERNAL_PROJECT_OPTIONAL_CMAKE_ARGS
    -DCMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=ON
    )
endif()

set(EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS)

# Force Python build to "Release".
if(CMAKE_CONFIGURATION_TYPES)
  set(SAVED_CMAKE_CFG_INTDIR ${CMAKE_CFG_INTDIR})
  set(CMAKE_CFG_INTDIR "Release")
else()
  list(APPEND EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=Release)
endif()

ExternalProject_Add(python
  GIT_REPOSITORY    "https://github.com/python-cmake-buildsystem/python-cmake-buildsystem.git"
  GIT_TAG           "ed5f9bcee540e47f82fa17f8360b820591aa6d66"
  PREFIX            ${PYTHON_SOURCES_DIR}
  UPDATE_COMMAND    ""
  CMAKE_CACHE_ARGS
    -DCMAKE_INSTALL_PREFIX:PATH=${PYTHON_INSTALL_DIR}
    -DBUILD_SHARED:BOOL=OFF
    -DBUILD_STATIC:BOOL=ON
    -DUSE_SYSTEM_LIBRARIES:BOOL=OFF
    -DZLIB_ROOT:FILEPATH=${ZLIB_ROOT}
    -DZLIB_INCLUDE_DIR:PATH=${ZLIB_INCLUDE_DIR}
    -DZLIB_LIBRARY:FILEPATH=${ZLIB_LIBRARIES}
    -DDOWNLOAD_SOURCES:BOOL=ON
    -DINSTALL_WINDOWS_TRADITIONAL:BOOL=OFF
    ${EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS}
  ${EXTERNAL_PROJECT_OPTIONAL_CMAKE_ARGS}
  DEPENDS zlib
)

set(_python_DIR ${PYTHON_INSTALL_DIR})

if(UNIX)
  set(_python_IMPORT_SUFFIX so)
  if(APPLE)
    set(_python_IMPORT_SUFFIX dylib)
  endif()
  set(PYTHON_INCLUDE_DIR "${PYTHON_INSTALL_DIR}/include/python2.7" CACHE PATH "Python include dir" FORCE)
  set(PYTHON_LIBRARY "${PYTHON_INSTALL_DIR}/lib/libpython2.7.${_python_IMPORT_SUFFIX}" CACHE FILEPATH "Python library" FORCE)
  set(PYTHON_EXECUTABLE ${PYTHON_INSTALL_DIR}/bin/python CACHE FILEPATH "Python executable" FORCE)
  set(PY_SITE_PACKAGES_PATH "${PYTHON_INSTALL_DIR}/lib/python2.7/site-packages" CACHE PATH "Python site-packages path" FORCE)
elseif(WIN32)
  set(PYTHON_INCLUDE_DIR "${PYTHON_INSTALL_DIR}/include" CACHE PATH "Python include dir" FORCE)
  set(PYTHON_LIBRARY "${PYTHON_INSTALL_DIR}/libs/python27.lib" CACHE FILEPATH "Python library" FORCE)
  set(PYTHON_EXECUTABLE "${PYTHON_INSTALL_DIR}/bin/python.exe" CACHE FILEPATH "Python executable" FORCE)
  set(PY_SITE_PACKAGES_PATH "${PYTHON_INSTALL_DIR}/Lib/site-packages" CACHE PATH "Python site-packages path" FORCE)
else()
  message(FATAL_ERROR "Unknown system !")
endif()
