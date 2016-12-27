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

FIND_PACKAGE(PythonLibs 2.7)
FIND_PACKAGE(PythonInterp 2.7)

IF((NOT ${PYTHONINTERP_FOUND}) OR (NOT ${PYTHONLIBS_FOUND}))

  INCLUDE(ExternalProject)

  SET(PYTHON_SOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/Python)
  SET(PYTHON_INSTALL_DIR ${PROJECT_BINARY_DIR}/Python)

  IF(MSVC)
    LIST(APPEND EXTERNAL_PROJECT_OPTIONAL_ARGS
      PATCH_COMMAND ${CMAKE_COMMAND}
        -DPYTHON_SRC_DIR:PATH=${_python_SOURCE_DIR}
        -P ${CMAKE_CURRENT_LIST_DIR}/PythonPatch.cmake
      )
  ENDIF()

  IF(APPLE)
    LIST(APPEND EXTERNAL_PROJECT_OPTIONAL_CMAKE_ARGS
      -DCMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=ON
      )
  ENDIF()

  SET(EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS)

  # Force Python build to "Release".
  IF(CMAKE_CONFIGURATION_TYPES)
    SET(SAVED_CMAKE_CFG_INTDIR ${CMAKE_CFG_INTDIR})
    SET(CMAKE_CFG_INTDIR "Release")
  ELSE()
    LIST(APPEND EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS
      -DCMAKE_BUILD_TYPE:STRING=Release)
  ENDIF()

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

  SET(_python_DIR ${PYTHON_INSTALL_DIR})

  IF(UNIX)
    SET(_python_IMPORT_SUFFIX a)
    IF(APPLE)
      SET(_python_IMPORT_SUFFIX lib)
    ENDIF()
    SET(PYTHON_INCLUDE_DIR "${PYTHON_INSTALL_DIR}/include/python2.7" CACHE PATH "Python include dir" FORCE)
    SET(PYTHON_LIBRARIES "${PYTHON_INSTALL_DIR}/lib/libpython2.7.${_python_IMPORT_SUFFIX}" CACHE FILEPATH "Python library" FORCE)
    SET(PYTHON_EXECUTABLE ${PYTHON_INSTALL_DIR}/bin/python CACHE FILEPATH "Python executable" FORCE)
    SET(PY_SITE_PACKAGES_PATH "${PYTHON_INSTALL_DIR}/lib/python2.7/site-packages" CACHE PATH "Python site-packages path" FORCE)
  ELSEIF(WIN32)
    SET(PYTHON_INCLUDE_DIR "${PYTHON_INSTALL_DIR}/include" CACHE PATH "Python include dir" FORCE)
    SET(PYTHON_LIBRARIES "${PYTHON_INSTALL_DIR}/libs/python27.lib" CACHE FILEPATH "Python library" FORCE)
    SET(PYTHON_EXECUTABLE "${PYTHON_INSTALL_DIR}/bin/python.exe" CACHE FILEPATH "Python executable" FORCE)
    SET(PY_SITE_PACKAGES_PATH "${PYTHON_INSTALL_DIR}/Lib/site-packages" CACHE PATH "Python site-packages path" FORCE)
  ELSE()
    MESSAGE(FATAL_ERROR "Unknown system !")
  ENDIF()

LIST(APPEND external_project_dependencies python)

ENDIF()

INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIR})

