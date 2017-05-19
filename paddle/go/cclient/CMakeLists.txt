cmake_minimum_required(VERSION 3.0)

if(GTEST_INCLUDE_DIR AND GTEST_LIBRARIES)
  message("-- Found gtest (include: ${GTEST_INCLUDE_DIR}, library: ${GTEST_LIBRARIES})")
else()	
  # find #include <majel/xx.h>
  get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
  include_directories(${PARENT_DIR})

  # find cmake directory modules
  get_filename_component(PARENT_DIR ${PARENT_DIR} DIRECTORY)
  get_filename_component(PARENT_DIR ${PARENT_DIR} DIRECTORY)

  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PARENT_DIR}/cmake")

  # enable c++11
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

  # enable gtest
  set(THIRD_PARTY_PATH ./third_party)
  set(WITH_TESTING ON)
  include(external/gtest)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(cxx_go CXX C Go)

include(cmake/golang.cmake)
include(cmake/flags.cmake)

ExternalGoProject_Add(pserver github.com/PaddlePaddle/Paddle/paddle/go/pserver)
add_go_library(client STATIC pserver)
add_subdirectory(test)
