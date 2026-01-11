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

set(Python_FIND_VIRTUALENV FIRST)

# Find Python with minimum PY_VERSION specified or will raise error!
find_package(
  Python ${PY_VERSION} EXACT
  COMPONENTS Interpreter Development
  REQUIRED)

set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
set(PYTHON_INCLUDE_DIR ${Python_INCLUDE_DIRS})
set(PYTHON_LIBRARIES ${Python_LIBRARIES})

list(GET Python_LIBRARIES 0 PYTHON_LIBRARY)

# Fixme: Maybe find a static library. Get SHARED/STATIC by FIND_PACKAGE.
add_library(python SHARED IMPORTED GLOBAL)
set_property(TARGET python PROPERTY IMPORTED_LOCATION ${PYTHON_LIBRARIES})

set(py_env "")
if(Python_FOUND)
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
