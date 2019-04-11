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

FIND_PACKAGE(PythonInterp ${PY_VERSION} REQUIRED)
FIND_PACKAGE(PythonLibs ${PY_VERSION} REQUIRED)

if(WIN32)
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
"from distutils import sysconfig as s;import sys;import struct;
print(sys.prefix);
print(s.get_config_var('LDVERSION') or s.get_config_var('VERSION'));
"
            RESULT_VARIABLE _PYTHON_SUCCESS
            OUTPUT_VARIABLE _PYTHON_VALUES
            ERROR_VARIABLE _PYTHON_ERROR_VALUE)

    if(NOT _PYTHON_SUCCESS MATCHES 0)
        set(PYTHONLIBS_FOUND FALSE)
        return()
    endif()

    # Convert the process output into a list
    string(REGEX REPLACE ";" "\\\\;" _PYTHON_VALUES ${_PYTHON_VALUES})
    string(REGEX REPLACE "\n" ";" _PYTHON_VALUES ${_PYTHON_VALUES})
    list(GET _PYTHON_VALUES 0 PYTHON_PREFIX)
    list(GET _PYTHON_VALUES 1 PYTHON_LIBRARY_SUFFIX)

    # Make sure all directory separators are '/'
    string(REGEX REPLACE "\\\\" "/" PYTHON_PREFIX ${PYTHON_PREFIX})

    set(PYTHON_LIBRARY
            "${PYTHON_PREFIX}/libs/Python${PYTHON_LIBRARY_SUFFIX}.lib")

    # when run in a venv, PYTHON_PREFIX points to it. But the libraries remain in the
    # original python installation. They may be found relative to PYTHON_INCLUDE_DIR.
    if(NOT EXISTS "${PYTHON_LIBRARY}")
        get_filename_component(_PYTHON_ROOT ${PYTHON_INCLUDE_DIR} DIRECTORY)
        set(PYTHON_LIBRARY
                "${_PYTHON_ROOT}/libs/Python${PYTHON_LIBRARY_SUFFIX}.lib")
    endif()

    # raise an error if the python libs are still not found.
    if(NOT EXISTS "${PYTHON_LIBRARY}")
        message(FATAL_ERROR "Python libraries not found")
    endif()
    SET(PYTHON_LIBRARIES "${PYTHON_LIBRARY}")
endif(WIN32)

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
