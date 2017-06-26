cmake_minimum_required(VERSION 3.0)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
get_filename_component(PARENT_DIR ${PARENT_DIR} DIRECTORY)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PARENT_DIR}/cmake")

project(cxx_go C Go)

include(golang)
include(flags)

cc_library(paddle_go_optimizer DEPS paddle_optimizer paddle_proto glog gflags)

go_library(paddle_pserver_cclient STATIC)
if(WITH_TESTING)
  add_subdirectory(test)
endif()
