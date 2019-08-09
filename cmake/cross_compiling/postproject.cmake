# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if (ANDROID)
    include(cross_compiling/findar)
endif()

if(ARMLINUX)
    if(ARMLINUX_ARCH_ABI STREQUAL "armv8")
        set(CMAKE_CXX_FLAGS "-march=armv8-a ${CMAKE_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "-march=armv8-a ${CMAKE_C_FLAGS}")
        message(STATUS "NEON is enabled on arm64-v8a")
    endif()

    if(ARMLINUX_ARCH_ABI STREQUAL "armv7")
        set(CMAKE_CXX_FLAGS "-march=armv7-a -mfloat-abi=softfp -mfpu=neon-vfpv4 ${CMAKE_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "-march=armv7-a -mfloat-abi=softfp -mfpu=neon-vfpv4 ${CMAKE_C_FLAGS}")
        message(STATUS "NEON is enabled on arm-v7a with softfp")
    endif()

    if(ARMLINUX_ARCH_ABI STREQUAL "armv7hf")
        set(CMAKE_CXX_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 ${CMAKE_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 ${CMAKE_C_FLAGS}" )
        message(STATUS "NEON is enabled on arm-v7a with hard float")
    endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(LITE_WITH_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OPENMP_FOUND OR OpenMP_CXX_FOUND)
        add_definitions(-DARM_WITH_OMP)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        message(STATUS "Found OpenMP ${OpenMP_VERSION} ${OpenMP_CXX_VERSION}")
        message(STATUS "OpenMP C flags:  ${OpenMP_C_FLAGS}")
        message(STATUS "OpenMP CXX flags:  ${OpenMP_CXX_FLAGS}")
        message(STATUS "OpenMP OpenMP_CXX_LIB_NAMES:  ${OpenMP_CXX_LIB_NAMES}")
        message(STATUS "OpenMP OpenMP_CXX_LIBRARIES:  ${OpenMP_CXX_LIBRARIES}")
    else()
        message(FATAL_ERROR "Could not found OpenMP!")
    endif()
endif()

