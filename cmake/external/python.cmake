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

IF(NOT WITH_PYTHON)
    return()
ENDIF()

INCLUDE(python_module)

FIND_PACKAGE(PythonInterp 2.7)
FIND_PACKAGE(PythonLibs 2.7)
# Fixme: Maybe find a static library. Get SHARED/STATIC by FIND_PACKAGE.
ADD_LIBRARY(python SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET python PROPERTY IMPORTED_LOCATION ${PYTHON_LIBRARIES})

SET(py_env "")
IF(PYTHONINTERP_FOUND)
    find_python_module(pip REQUIRED)
    find_python_module(numpy REQUIRED)
    find_python_module(wheel REQUIRED)
    find_python_module(google.protobuf REQUIRED)
    FIND_PACKAGE(NumPy REQUIRED)
    IF(${PY_GOOGLE.PROTOBUF_VERSION} AND ${PY_GOOGLE.PROTOBUF_VERSION} VERSION_LESS "3.0.0")
        MESSAGE(FATAL_ERROR "Found Python Protobuf ${PY_GOOGLE.PROTOBUF_VERSION} < 3.0.0, "
        "please use pip to upgrade protobuf. pip install -U protobuf")
    ENDIF()
ENDIF(PYTHONINTERP_FOUND)

INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${PYTHON_NUMPY_INCLUDE_DIR})
