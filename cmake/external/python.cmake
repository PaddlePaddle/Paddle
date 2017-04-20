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
INCLUDE(python_module)

FIND_PACKAGE(PythonInterp 2.7)
FIND_PACKAGE(PythonLibs 2.7)

SET(py_env "")

IF(PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND)
    find_python_module(pip REQUIRED)
    find_python_module(numpy REQUIRED)
    find_python_module(wheel REQUIRED)
    find_python_module(google.protobuf REQUIRED)
    FIND_PACKAGE(NumPy REQUIRED)
    IF(${PY_GOOGLE.PROTOBUF_VERSION} AND ${PY_GOOGLE.PROTOBUF_VERSION} VERSION_LESS "3.0.0")
        MESSAGE(FATAL_ERROR "Found Python Protobuf ${PY_GOOGLE.PROTOBUF_VERSION} < 3.0.0, "
        "please use pip to upgrade protobuf. pip install -U protobuf")
    ENDIF()
ELSE(PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND)
    MESSAGE(FATAL_ERROR "Please install python 2.7 before building PaddlePaddle.")
    ##################################### PYTHON ########################################
    SET(PYTHON_SOURCES_DIR ${THIRD_PARTY_PATH}/python)
    SET(PYTHON_INSTALL_DIR ${THIRD_PARTY_PATH}/install/python)
    SET(_python_DIR ${PYTHON_INSTALL_DIR})

    IF(UNIX)
        SET(PYTHON_FOUND ON)
        SET(PYTHON_INCLUDE_DIR "${PYTHON_INSTALL_DIR}/include/python2.7" CACHE PATH "Python include dir" FORCE)
        SET(PYTHON_LIBRARIES "${PYTHON_INSTALL_DIR}/lib/libpython2.7.a" CACHE FILEPATH "Python library" FORCE)
        SET(PYTHON_EXECUTABLE ${PYTHON_INSTALL_DIR}/bin/python CACHE FILEPATH "Python executable" FORCE)
        SET(PY_SITE_PACKAGES_PATH "${PYTHON_INSTALL_DIR}/lib/python2.7/site-packages" CACHE PATH "Python site-packages path" FORCE)
    ELSEIF(WIN32)
        SET(PYTHON_FOUND ON)
        SET(PYTHON_INCLUDE_DIR "${PYTHON_INSTALL_DIR}/include" CACHE PATH "Python include dir" FORCE)
        SET(PYTHON_LIBRARIES "${PYTHON_INSTALL_DIR}/libs/python27.lib" CACHE FILEPATH "Python library" FORCE)
        SET(PYTHON_EXECUTABLE "${PYTHON_INSTALL_DIR}/bin/python.exe" CACHE FILEPATH "Python executable" FORCE)
        SET(PY_SITE_PACKAGES_PATH "${PYTHON_INSTALL_DIR}/Lib/site-packages" CACHE PATH "Python site-packages path" FORCE)
    ELSE()
        MESSAGE(FATAL_ERROR "Unknown system !")
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
            -DCMAKE_BUILD_TYPE:STRING=Release
            )
    ENDIF()

    ExternalProject_Add(python
        ${EXTERNAL_PROJECT_LOG_ARGS}
        GIT_REPOSITORY    "https://github.com/python-cmake-buildsystem/python-cmake-buildsystem.git"
        PREFIX            ${PYTHON_SOURCES_DIR}
        UPDATE_COMMAND    ""
        CMAKE_ARGS        -DPYTHON_VERSION=2.7.12
        CMAKE_ARGS        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        CMAKE_ARGS        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        CMAKE_CACHE_ARGS
            -DCMAKE_INSTALL_PREFIX:PATH=${PYTHON_INSTALL_DIR}
            -DBUILD_LIBPYTHON_SHARED:BOOL=OFF
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

    SET(py_env
        PATH=${PYTHON_INSTALL_DIR}/bin
        PYTHONHOME=${PYTHON_INSTALL_DIR}
        PYTHONPATH=${PYTHON_INSTALL_DIR}/lib:${PYTHON_INSTALL_DIR}/lib/python2.7:${PY_SITE_PACKAGES_PATH})
    ####################################################################################

    ##################################### SETUPTOOLS ###################################
    SET(SETUPTOOLS_SOURCES_DIR ${PYTHON_SOURCES_DIR}/setuptools)
    ExternalProject_Add(setuptools
        ${EXTERNAL_PROJECT_LOG_ARGS}
        PREFIX              ${SETUPTOOLS_SOURCES_DIR}
        URL                 "https://pypi.python.org/packages/source/s/setuptools/setuptools-18.3.2.tar.gz"
        BUILD_IN_SOURCE     1
        PATCH_COMMAND       ""
        UPDATE_COMMAND      ""
        CONFIGURE_COMMAND   ""
        INSTALL_COMMAND     ""
        BUILD_COMMAND       env ${py_env} ${PYTHON_EXECUTABLE} setup.py install
        DEPENDS             python zlib
    )
    #####################################################################################

    ##################################### SIX ###########################################
    SET(SIX_SOURCES_DIR ${PYTHON_SOURCES_DIR}/six)
    ExternalProject_Add(six
        ${EXTERNAL_PROJECT_LOG_ARGS}
        PREFIX              ${SIX_SOURCES_DIR}
        URL                 https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz
        BUILD_IN_SOURCE     1
        PATCH_COMMAND       ""
        UPDATE_COMMAND      ""
        CONFIGURE_COMMAND   ""
        INSTALL_COMMAND     ""
        BUILD_COMMAND       env ${py_env} ${PYTHON_EXECUTABLE} setup.py install
        DEPENDS             python setuptools
    )
    #####################################################################################

    ##################################### CYTHON ########################################
    SET(CYTHON_SOURCES_DIR ${PYTHON_SOURCES_DIR}/cython)
    ExternalProject_Add(cython
        ${EXTERNAL_PROJECT_LOG_ARGS}
        PREFIX                ${CYTHON_SOURCES_DIR}
        URL                   https://github.com/cython/cython/archive/0.25.2.tar.gz
        GIT_TAG               0.25.2
        BUILD_IN_SOURCE       1
        CONFIGURE_COMMAND     ""
        PATCH_COMMAND         ""
        UPDATE_COMMAND        ""
        INSTALL_COMMAND       ""
        BUILD_COMMAND         env ${py_env} ${PYTHON_EXECUTABLE} setup.py install
        DEPENDS               python
    )
    ####################################################################################

    ##################################### NUMPY ########################################
    SET(NUMPY_SOURCES_DIR ${PYTHON_SOURCES_DIR}/numpy)
    SET(NUMPY_TAG_VERSION "v1.11.3")
    SET(NUMPY_VERSION "1.11.3")

    SET(EGG_NAME "")
    SET(PYTHON_NUMPY_INCLUDE_DIR "")
    IF(WIN32)
        SET(EGG_NAME "numpy-${NUMPY_VERSION}-py2.7-${HOST_SYSTEM}.egg")
    ELSE(WIN32)
        IF(APPLE)
            SET(EGG_NAME "numpy-${NUMPY_VERSION}-py2.7-${HOST_SYSTEM}-${MACOS_VERSION}")
        ELSE(APPLE)
            SET(EGG_NAME "numpy-${NUMPY_VERSION}-py2.7-linux")
            SET(EGG_NAME "numpy-${NUMPY_VERSION}-py2.7-linux")
        ENDIF(APPLE)

        FOREACH(suffix x86_64 intel fat64 fat32 universal)
            LIST(APPEND PYTHON_NUMPY_INCLUDE_DIR ${PY_SITE_PACKAGES_PATH}/${EGG_NAME}-${suffix}.egg/numpy/core/include)
        ENDFOREACH()
    ENDIF(WIN32)

    ExternalProject_Add(numpy
        ${EXTERNAL_PROJECT_LOG_ARGS}
        GIT_REPOSITORY      https://github.com/numpy/numpy.git
        GIT_TAG             ${NUMPY_TAG_VERSION}
        CONFIGURE_COMMAND   ""
        UPDATE_COMMAND      ""
        PREFIX              ${NUMPY_SOURCES_DIR}
        BUILD_COMMAND       env ${py_env} ${PYTHON_EXECUTABLE} setup.py build
        INSTALL_COMMAND     env ${py_env} ${PYTHON_EXECUTABLE} setup.py install
        BUILD_IN_SOURCE     1
        DEPENDS             python setuptools cython
    )
    ####################################################################################

    ##################################### WHEEL ########################################
    SET(WHEEL_SOURCES_DIR ${PYTHON_SOURCES_DIR}/wheel)
    ExternalProject_Add(wheel
        ${EXTERNAL_PROJECT_LOG_ARGS}
        URL                 https://pypi.python.org/packages/source/w/wheel/wheel-0.29.0.tar.gz
        PREFIX              ${WHEEL_SOURCES_DIR}
        CONFIGURE_COMMAND   ""
        UPDATE_COMMAND      ""
        BUILD_COMMAND       ""
        INSTALL_COMMAND     env ${py_env} ${PYTHON_EXECUTABLE} setup.py install
        BUILD_IN_SOURCE     1
        DEPENDS             python setuptools
    )
    ####################################################################################

    ################################### PROTOBUF #######################################
    SET(PY_PROTOBUF_SOURCES_DIR ${PYTHON_SOURCES_DIR}/protobuf)
    ExternalProject_Add(python-protobuf
        ${EXTERNAL_PROJECT_LOG_ARGS}
        URL                   https://pypi.python.org/packages/e0/b0/0a1b364fe8a7d177b4b7d4dca5b798500dc57a7273b93cca73931b305a6a/protobuf-3.1.0.post1.tar.gz
        URL_MD5               38b5fb160c768d2f8444d0c6d637ff91
        PREFIX                ${PY_PROTOBUF_SOURCES_DIR}
        BUILD_IN_SOURCE       1
        PATCH_COMMAND         ""
        CONFIGURE_COMMAND     ""
        BUILD_COMMAND         env ${py_env} ${PYTHON_EXECUTABLE} setup.py build
        INSTALL_COMMAND       env ${py_env} ${PYTHON_EXECUTABLE} setup.py install
        DEPENDS               python setuptools six
    )
    ####################################################################################

    LIST(APPEND external_project_dependencies python setuptools six cython wheel python-protobuf numpy)

ENDIF(PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND)

IF(WITH_PYTHON)
    INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIR})
    INCLUDE_DIRECTORIES(${PYTHON_NUMPY_INCLUDE_DIR})
ELSE()
    SET(PYTHON_LIBRARIES "")
ENDIF()
