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

if(USE_OPENMP STREQUAL "gnu")
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions(-DCINN_USE_OPENMP)
    set(WITH_OPENMP ON)
    message(STATUS "Build with OpenMP ${OpenMP_CXX_LIBRARIES}")
    message(STATUS "CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS})
  else()
    set(WITH_OPENMP OFF)
  endif()
elseif(USE_OPENMP STREQUAL "intel")
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS})
    add_definitions(-DCINN_USE_OPENMP)
    set(WITH_OPENMP ON)
    message(STATUS "Build with OpenMP " ${MKLML_IOMP_LIB})
  else()
    set(WITH_OPENMP OFF)
  endif()
endif()
