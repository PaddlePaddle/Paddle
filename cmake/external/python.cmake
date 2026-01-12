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

include(python_module)

check_py_version(${PY_VERSION})

if(DEFINED PYTHON_EXECUTABLE AND NOT DEFINED Python_EXECUTABLE)
  set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

# Debug: Print PATH information
message(STATUS "=== Python Debug Information ===")
message(STATUS "PY_VERSION: ${PY_VERSION}")
message(STATUS "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
if(UNIX AND NOT APPLE)
  execute_process(
    COMMAND sh -c "echo $PATH"
    OUTPUT_VARIABLE PATH_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(PATH_VALUE)
    message(STATUS "PATH: ${PATH_VALUE}")
  endif()
  execute_process(
    COMMAND sh -c "which python 2>/dev/null || true"
    OUTPUT_VARIABLE WHICH_PYTHON
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(WHICH_PYTHON)
    message(STATUS "which python: ${WHICH_PYTHON}")
  endif()
  execute_process(
    COMMAND sh -c "which python3 2>/dev/null || true"
    OUTPUT_VARIABLE WHICH_PYTHON3
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(WHICH_PYTHON3)
    message(STATUS "which python3: ${WHICH_PYTHON3}")
  endif()
  execute_process(
    COMMAND sh -c "which python${PY_VERSION} 2>/dev/null || true"
    OUTPUT_VARIABLE WHICH_PYTHON_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(WHICH_PYTHON_VERSION)
    message(STATUS "which python${PY_VERSION}: ${WHICH_PYTHON_VERSION}")
  endif()
endif()
message(STATUS "=== End Python Debug Information ===")

# Temporarily revert to old FindPythonInterp/FindPythonLibs for debugging
find_package(PythonInterp ${PY_VERSION} REQUIRED)
find_package(PythonLibs ${PY_VERSION} REQUIRED)

list(GET PYTHON_LIBRARIES 0 PYTHON_LIBRARY)

# Fixme: Maybe find a static library. Get SHARED/STATIC by FIND_PACKAGE.
add_library(python SHARED IMPORTED GLOBAL)
set_property(TARGET python PROPERTY IMPORTED_LOCATION ${PYTHON_LIBRARIES})

set(py_env "")
if(PYTHONINTERP_FOUND)
  find_python_module(pip REQUIRED)
  find_python_module(numpy REQUIRED)
  find_python_module(wheel REQUIRED)
  find_python_module(google.protobuf REQUIRED)
  find_package(NumPy REQUIRED)
  if(${PY_GOOGLE.PROTOBUF_VERSION} AND ${PY_GOOGLE.PROTOBUF_VERSION}
                                       VERSION_LESS "3.0.0")
    message(
      FATAL_ERROR
        "Found Python Protobuf ${PY_GOOGLE.PROTOBUF_VERSION} < 3.0.0, "
        "please use pip to upgrade protobuf. pip install -U protobuf")
  endif()
endif()

include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${PYTHON_NUMPY_INCLUDE_DIR})
