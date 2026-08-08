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

foreach(name EXECUTABLE INCLUDE_DIR LIBRARY)
  if(PYTHON_${name} AND NOT Python_${name})
    set(Python_${name} "${PYTHON_${name}}")
  endif()
endforeach()
if(PYTHON_NUMPY_INCLUDE_DIR AND NOT Python_NumPy_INCLUDE_DIR)
  set(Python_NumPy_INCLUDE_DIR "${PYTHON_NUMPY_INCLUDE_DIR}")
endif()

set(Python_FIND_STRATEGY LOCATION)
set(Python_FIND_VIRTUALENV FIRST)
if(POLICY CMP0190)
  cmake_policy(SET CMP0190 NEW)
endif()
if(POLICY CMP0201)
  cmake_policy(SET CMP0201 NEW)
endif()
find_package(Python ${PY_VERSION} REQUIRED COMPONENTS Interpreter Development
                                                      NumPy)

# Transitional outputs keep dependent stacked PRs buildable while call sites
# migrate to modern result variables and imported targets.
set(PYTHON_EXECUTABLE "${Python_EXECUTABLE}")
set(PYTHON_INCLUDE_DIR "${Python_INCLUDE_DIRS}")
set(PYTHON_INCLUDE_DIRS "${Python_INCLUDE_DIRS}")
set(PYTHON_LIBRARY "${Python_LIBRARY}")
set(PYTHON_LIBRARIES "${Python_LIBRARIES}")
set(PYTHON_NUMPY_INCLUDE_DIR "${Python_NumPy_INCLUDE_DIRS}")
set(PYTHONINTERP_FOUND "${Python_Interpreter_FOUND}")

add_library(python INTERFACE IMPORTED GLOBAL)
target_link_libraries(python INTERFACE Python::Module)

set(py_env "")
if(Python_Interpreter_FOUND)
  find_python_module(pip REQUIRED)
  find_python_module(numpy REQUIRED)
  find_python_module(wheel REQUIRED)
  find_python_module(google.protobuf REQUIRED)
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
