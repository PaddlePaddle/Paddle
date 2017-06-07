cmake_minimum_required(VERSION 3.0)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
get_filename_component(PARENT_DIR ${PARENT_DIR} DIRECTORY)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PARENT_DIR}/cmake")

project(cxx_go C Go)

include(golang)
include(flags)

go_library(paddle_pserver_cclient STATIC)

if(PROJ_ROOT)
  add_custom_command(OUTPUT ${PROJ_ROOT}/paddle/trainer/libpaddle_pserver_cclient.a
          COMMAND cp ${CMAKE_BINARY_DIR}/go/pserver/cclient/libpaddle_pserver_cclient.h ${PROJ_ROOT}/paddle/trainer/
          COMMAND cp ${CMAKE_BINARY_DIR}/go/pserver/cclient/libpaddle_pserver_cclient.a ${PROJ_ROOT}/paddle/trainer/
          WORKING_DIRECTORY ${PROJ_ROOT}/paddle
          DEPENDS paddle_pserver_cclient)
  add_custom_target(paddle_pserver_cclient_lib ALL DEPENDS ${PROJ_ROOT}/paddle/trainer/libpaddle_pserver_cclient.a)
endif(PROJ_ROOT)

add_subdirectory(test)
