cmake_minimum_required(VERSION 3.0)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
get_filename_component(PARENT_DIR ${PARENT_DIR} DIRECTORY)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PARENT_DIR}/cmake")

project(cxx_go C Go)

include(golang)
include(flags)

set(MASTER_LIB_NAME "paddle_master")
go_library(${MASTER_LIB_NAME} SHARED)

if(PROJ_ROOT)
  add_custom_command(OUTPUT ${PROJ_ROOT}/python/paddle/v2/master/lib${MASTER_LIB_NAME}.so
    COMMAND rm ${CMAKE_CURRENT_BINARY_DIR}/lib${MASTER_LIB_NAME}.h
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/lib${MASTER_LIB_NAME}.so ${PROJ_ROOT}/python/paddle/v2/master/
    DEPENDS ${MASTER_LIB_NAME})
  add_custom_target(paddle_master_shared ALL DEPENDS ${PROJ_ROOT}/python/paddle/v2/master/lib${MASTER_LIB_NAME}.so)
endif(PROJ_ROOT)
