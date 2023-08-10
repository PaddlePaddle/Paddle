cmake_minimum_required(VERSION 3.10)

project(example LANGUAGES CXX)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

if(NOT DEFINED XDNN_PATH)
  set(XDNN_PATH $ENV{XDNN_PATH})
endif()
if(NOT DEFINED XRE_PATH)
  set(XRE_PATH $ENV{XRE_PATH})
endif()
if(NOT IS_DIRECTORY ${XDNN_PATH})
  message(
    FATAL_ERROR
      "XDNN_PATH not set, or directory ${XDNN_PATH} not found, please export XDNN_PATH=<path_to_xdnn>."
  )
endif()
if(NOT IS_DIRECTORY ${XRE_PATH})
  message(
    FATAL_ERROR
      "XRE_PATH not set, or directory ${XRE_PATH} not found, please export XRE_PATH=<path_to_xre>."
  )
endif()
set(XDNN_INC_DIR ${XDNN_PATH}/include)
set(XDNN_LIB_DIR ${XDNN_PATH}/so)
set(XRE_INC_DIR ${XRE_PATH}/include)
set(XRE_LIB_DIR ${XRE_PATH}/so)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,--allow-shlib-undefined")

include_directories(${XDNN_INC_DIR})
include_directories(${XRE_INC_DIR})
link_directories(${XDNN_LIB_DIR})
link_directories(${XRE_LIB_DIR})
set(DEPS ${DEPS} xpurt xpuapi)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
if(NOT DEFINED LINK_TYPE)
  set(LINK_TYPE $ENV{LINK_TYPE})
endif()
if(LINK_TYPE STREQUAL "static")
  set(DEPS ${DEPS} ${CMAKE_CURRENT_SOURCE_DIR}/../build/libxpuplugin.a)
elseif(LINK_TYPE STREQUAL "shared")
  link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../build)
  set(DEPS ${DEPS} xpuplugin)
else()
  message(
    FATAL_ERROR
      "Unknown LINK_TYPE ${LINK_TYPE}, only supports static or shared.")
  return()
endif()

add_executable(example example.cc)
target_link_libraries(example ${DEPS})
