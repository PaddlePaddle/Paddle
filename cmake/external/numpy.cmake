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

SET(NUMPY_SOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/numpy)
SET(NUMPY_INSTALL_DIR ${PROJECT_BINARY_DIR}/numpy)
set(NUMPY_VERSION "v1.11.3")

# setuptools
ExternalProject_Add(setuptools
    PREFIX              ${PYTHON_SOURCES_DIR}/setuptools
    URL                 http://pypi.python.org/packages/source/s/setuptools/setuptools-0.6c11.tar.gz
    URL_MD5             7df2a529a074f613b509fb44feefe74e
    BUILD_IN_SOURCE     1
    UPDATE_COMMAND      ""
    PATCH_COMMAND       ""
    CONFIGURE_COMMAND   ""
    INSTALL_COMMAND     ""
    BUILD_COMMAND       ${PYTHON_EXECUTABLE} setup.py install
    DEPENDS             python zlib
)

ExternalProject_Add(cython
  PREFIX                ${PYTHON_SOURCES_DIR}/cython
  GIT_REPOSITORY        https://github.com/cython/cython.git
  BUILD_IN_SOURCE       1
  CONFIGURE_COMMAND     ""
  UPDATE_COMMAND        ""
  PATCH_COMMAND         ""
  INSTALL_COMMAND       ""
  BUILD_COMMAND         ${PYTHON_EXECUTABLE} setup.py install
  DEPENDS               python
)

ExternalProject_Add(numpy
    GIT_REPOSITORY      https://github.com/numpy/numpy.git
    GIT_TAG             ${NUMPY_VERSION}
    CONFIGURE_COMMAND   ""
    UPDATE_COMMAND      ""
    PREFIX              ${NUMPY_SOURCES_DIR}
    BUILD_COMMAND       ${PYTHON_EXECUTABLE} setup.py build
    INSTALL_COMMAND     ${PYTHON_EXECUTABLE} setup.py install
    BUILD_IN_SOURCE     1
    DEPENDS python setuptools cython
)
