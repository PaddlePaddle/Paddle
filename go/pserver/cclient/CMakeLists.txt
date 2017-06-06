cmake_minimum_required(VERSION 3.0)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
get_filename_component(PARENT_DIR ${PARENT_DIR} DIRECTORY)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PARENT_DIR}/cmake")

project(cxx_go C Go)

include(golang)
include(flags)

add_custom_command(OUTPUT ${PROJ_ROOT}/paddle/trainer/libpaddle_cclient.a
        COMMAND cp ${CMAKE_BINARY_DIR}/go/pserver/cclient/libpaddle_cclient.h ${PROJ_ROOT}/paddle/trainer/
        COMMAND cp ${CMAKE_BINARY_DIR}/go/pserver/cclient/libpaddle_cclient.a ${PROJ_ROOT}/paddle/trainer/
        WORKING_DIRECTORY ${PROJ_ROOT}/paddle
        DEPENDS paddle_cclient)

add_custom_target(paddle_cclient_h ALL DEPENDS ${PROJ_ROOT}/paddle/trainer/libpaddle_cclient.a)

go_library(paddle_cclient STATIC)
add_subdirectory(test)
