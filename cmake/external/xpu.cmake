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

if(NOT WITH_XPU)
  return()
endif()

include(ExternalProject)
set(XPU_PROJECT "extern_xpu")
set(XPU_API_LIB_NAME "libxpuapi.so")
set(XPU_RT_LIB_NAME "libxpurt.so")
set(XPU_XFT_LIB_NAME "libxft.so")
set(XPU_XPTI_LIB_NAME "libxpti.so")

if(NOT DEFINED XPU_BASE_DATE)
  set(XPU_BASE_DATE "20230926")
endif()
set(XPU_XCCL_BASE_VERSION "1.0.53.6")
if(NOT DEFINED XPU_XFT_BASE_VERSION)
  set(XPU_XFT_BASE_VERSION "20230602")
endif()
set(XPU_XPTI_BASE_VERSION "0.0.1")

if(NOT DEFINED XPU_BASE_URL)
  set(XPU_BASE_URL_WITHOUT_DATE
      "https://baidu-kunlun-product.su.bcebos.com/KL-SDK/klsdk-dev")
  set(XPU_BASE_URL "${XPU_BASE_URL_WITHOUT_DATE}/${XPU_BASE_DATE}")
else()
  set(XPU_BASE_URL "${XPU_BASE_URL}")
endif()

set(XPU_XCCL_BASE_URL
    "https://klx-sdk-release-public.su.bcebos.com/xccl/release/${XPU_XCCL_BASE_VERSION}"
)

if(NOT XPU_XFT_BASE_URL)
  set(XPU_XFT_BASE_URL
      "https://klx-sdk-release-public.su.bcebos.com/xft/dev/${XPU_XFT_BASE_VERSION}"
  )
endif()

set(XPU_XPTI_BASE_URL
    "https://klx-sdk-release-public.su.bcebos.com/xpti/dev/${XPU_XPTI_BASE_VERSION}"
)

if(WITH_XCCL_RDMA)
  set(XPU_XCCL_PREFIX "xccl_rdma")
else()
  set(XPU_XCCL_PREFIX "xccl_socket")
endif()

if(WITH_AARCH64)
  set(XPU_XRE_DIR_NAME "xre-kylin_aarch64")
  set(XPU_XDNN_DIR_NAME "xdnn-kylin_aarch64")
  set(XPU_XCCL_DIR_NAME "${XPU_XCCL_PREFIX}-kylin_aarch64")
  set(XPU_XFT_DIR_NAME "") # TODO: xft has no kylin output at now.
elseif(WITH_SUNWAY)
  set(XPU_XRE_DIR_NAME "xre-deepin_sw6_64")
  set(XPU_XDNN_DIR_NAME "xdnn-deepin_sw6_64")
  set(XPU_XCCL_DIR_NAME "${XPU_XCCL_PREFIX}-deepin_sw6_64")
  set(XPU_XFT_DIR_NAME "") # TODO: xft has no deepin output at now.
elseif(WITH_BDCENTOS)
  set(XPU_XRE_DIR_NAME "xre-bdcentos_x86_64")
  set(XPU_XDNN_DIR_NAME "xdnn-bdcentos_x86_64")
  set(XPU_XCCL_DIR_NAME "${XPU_XCCL_PREFIX}-bdcentos_x86_64")
  set(XPU_XFT_DIR_NAME "xft_bdcentos6u3_x86_64_gcc82")
elseif(WITH_UBUNTU)
  set(XPU_XRE_DIR_NAME "xre-ubuntu_x86_64")
  set(XPU_XDNN_DIR_NAME "xdnn-ubuntu_x86_64")
  set(XPU_XCCL_DIR_NAME "${XPU_XCCL_PREFIX}-ubuntu_x86_64")
  set(XPU_XFT_DIR_NAME "xft_ubuntu1604_x86_64")
elseif(WITH_CENTOS)
  set(XPU_XRE_DIR_NAME "xre-centos7_x86_64")
  set(XPU_XDNN_DIR_NAME "xdnn-centos7_x86_64")
  set(XPU_XCCL_DIR_NAME "${XPU_XCCL_PREFIX}-bdcentos_x86_64")
  set(XPU_XFT_DIR_NAME "xft_bdcentos6u3_x86_64_gcc82")
else()
  set(XPU_XRE_DIR_NAME "xre-ubuntu_x86_64")
  set(XPU_XDNN_DIR_NAME "xdnn-ubuntu_x86_64")
  set(XPU_XCCL_DIR_NAME "${XPU_XCCL_PREFIX}-ubuntu_x86_64")
  set(XPU_XFT_DIR_NAME "xft_ubuntu1604_x86_64")
endif()
set(XPU_XPTI_DIR_NAME "xpti")

set(XPU_XRE_URL
    "${XPU_BASE_URL}/${XPU_XRE_DIR_NAME}.tar.gz"
    CACHE STRING "" FORCE)
set(XPU_XDNN_URL
    "${XPU_BASE_URL}/${XPU_XDNN_DIR_NAME}.tar.gz"
    CACHE STRING "" FORCE)
set(XPU_XCCL_URL
    "${XPU_XCCL_BASE_URL}/${XPU_XCCL_DIR_NAME}.tar.gz"
    CACHE STRING "" FORCE)
set(XPU_XFT_URL "${XPU_XFT_BASE_URL}/${XPU_XFT_DIR_NAME}.tar.gz")
set(XPU_XPTI_URL
    "${XPU_XPTI_BASE_URL}/${XPU_XPTI_DIR_NAME}.tar.gz"
    CACHE STRING "" FORCE)
set(XPU_XFT_GET_DEPENCE_URL
    "https://baidu-kunlun-public.su.bcebos.com/paddle_depence/get_xft_dependence.sh"
    CACHE STRING "" FORCE)

set(SNAPPY_PREFIX_DIR "${THIRD_PARTY_PATH}/xpu")
set(XPU_DOWNLOAD_DIR "${SNAPPY_PREFIX_DIR}/src/${XPU_PROJECT}")
set(XPU_INSTALL_DIR "${THIRD_PARTY_PATH}/install/xpu")
set(XPU_INC_DIR "${THIRD_PARTY_PATH}/install/xpu/include")
set(XPU_LIB_DIR "${THIRD_PARTY_PATH}/install/xpu/lib")

set(XPU_API_LIB "${XPU_LIB_DIR}/${XPU_API_LIB_NAME}")
set(XPU_RT_LIB "${XPU_LIB_DIR}/${XPU_RT_LIB_NAME}")

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${XPU_INSTALL_DIR}/lib")

file(
  WRITE ${XPU_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(XPU)\n" "cmake_minimum_required(VERSION 3.0)\n"
  "install(DIRECTORY xpu/include xpu/lib \n"
  "        DESTINATION ${XPU_INSTALL_DIR})\n")

if(WITH_XPU_BKCL)
  message(STATUS "Compile with XPU BKCL!")
  add_definitions(-DPADDLE_WITH_XPU_BKCL)

  set(XPU_BKCL_LIB_NAME "libbkcl.so")
  set(XPU_BKCL_LIB "${XPU_LIB_DIR}/${XPU_BKCL_LIB_NAME}")
  set(XPU_BKCL_INC_DIR "${THIRD_PARTY_PATH}/install/xpu/include")
  include_directories(${XPU_BKCL_INC_DIR})
endif()

ExternalProject_Add(
  ${XPU_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  PREFIX ${SNAPPY_PREFIX_DIR}
  DOWNLOAD_DIR ${XPU_DOWNLOAD_DIR}
  DOWNLOAD_COMMAND
    bash ${CMAKE_SOURCE_DIR}/tools/xpu/check_xpu_dependence.sh ${XPU_BASE_URL}
    ${XPU_XCCL_BASE_URL} && bash
    ${CMAKE_SOURCE_DIR}/tools/xpu/pack_paddle_depence.sh ${XPU_XRE_URL}
    ${XPU_XRE_DIR_NAME} ${XPU_XDNN_URL} ${XPU_XDNN_DIR_NAME} ${XPU_XCCL_URL}
    ${XPU_XCCL_DIR_NAME} && wget ${XPU_XFT_GET_DEPENCE_URL} && bash
    get_xft_dependence.sh ${XPU_XFT_URL} ${XPU_XFT_DIR_NAME} &&
    WITH_XPTI=${WITH_XPTI} bash
    ${CMAKE_SOURCE_DIR}/tools/xpu/get_xpti_dependence.sh ${XPU_XPTI_URL}
    ${XPU_XPTI_DIR_NAME}
  DOWNLOAD_NO_PROGRESS 1
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${XPU_INSTALL_ROOT}
  CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${XPU_INSTALL_ROOT}
  BUILD_BYPRODUCTS ${XPU_API_LIB}
  BUILD_BYPRODUCTS ${XPU_RT_LIB}
  BUILD_BYPRODUCTS ${XPU_BKCL_LIB})

include_directories(${XPU_INC_DIR})
add_library(shared_xpuapi SHARED IMPORTED GLOBAL)
set_property(TARGET shared_xpuapi PROPERTY IMPORTED_LOCATION "${XPU_API_LIB}")

# generate a static dummy target to track xpulib dependencies
# for cc_library(xxx SRCS xxx.c DEPS xpulib)
generate_dummy_static_lib(LIB_NAME "xpulib" GENERATOR "xpu.cmake")

target_link_libraries(xpulib ${XPU_API_LIB} ${XPU_RT_LIB})

if(WITH_XPU_XFT)
  message(STATUS "Compile with XPU XFT!")
  add_definitions(-DPADDLE_WITH_XPU_XFT)

  set(XPU_XFT_INC_DIR "${XPU_INC_DIR}/xft")
  include_directories(${XPU_XFT_INC_DIR})
  set(XPU_XFT_LIB "${XPU_LIB_DIR}/${XPU_XFT_LIB_NAME}")
endif()

if(WITH_XPTI)
  message(STATUS "Compile with XPU XPTI!")
  add_definitions(-DPADDLE_WITH_XPTI)
  set(XPU_XPTI_LIB "${XPU_LIB_DIR}/${XPU_XPTI_LIB_NAME}")
endif()

if(WITH_XPU_PLUGIN)
  message(STATUS "Compile with XPU PLUGIN!")
  add_definitions(-DPADDLE_WITH_XPU_PLUGIN)
  include_directories(${CMAKE_SOURCE_DIR}/paddle/phi/kernels/xpu/plugin/include)
endif()

if(WITH_XPU_BKCL AND WITH_XPU_XFT)
  target_link_libraries(xpulib ${XPU_API_LIB} ${XPU_RT_LIB} ${XPU_BKCL_LIB}
                        ${XPU_XFT_LIB})
elseif(WITH_XPU_BKCL)
  target_link_libraries(xpulib ${XPU_API_LIB} ${XPU_RT_LIB} ${XPU_BKCL_LIB})
elseif(WITH_XPU_XFT)
  target_link_libraries(xpulib ${XPU_API_LIB} ${XPU_RT_LIB} ${XPU_XFT_LIB})
else()
  target_link_libraries(xpulib ${XPU_API_LIB} ${XPU_RT_LIB})
endif()

if(WITH_XPTI)
  target_link_libraries(xpulib ${XPU_XPTI_LIB})
endif()

add_dependencies(xpulib ${XPU_PROJECT})

# Ensure that xpu/api.h can be included without dependency errors.
file(
  GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/.xpu_headers_dummy.cc
  CONTENT "")
add_library(xpu_headers_dummy STATIC
            ${CMAKE_CURRENT_BINARY_DIR}/.xpu_headers_dummy.cc)
add_dependencies(xpu_headers_dummy extern_xpu)
link_libraries(xpu_headers_dummy)
