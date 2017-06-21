cmake_minimum_required(VERSION 3.0)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
get_filename_component(PARENT_DIR ${PARENT_DIR} DIRECTORY)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PARENT_DIR}/cmake")

project(cxx_go C Go)

include(golang)
include(flags)

go_library(paddle_trainer STATIC)

if(PROJ_ROOT)
  add_custom_command(OUTPUT ${PROJ_ROOT}/paddle/trainer/libpaddle_trainer.a
          COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/libpaddle_trainer.h ${PROJ_ROOT}/paddle/trainer/
          COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/libpaddle_trainer.a ${PROJ_ROOT}/paddle/trainer/
          WORKING_DIRECTORY ${PROJ_ROOT}/paddle
          DEPENDS paddle_trainer)
  add_custom_target(paddle_trainer_lib ALL DEPENDS ${PROJ_ROOT}/paddle/trainer/libpaddle_trainer.a)
endif(PROJ_ROOT)

add_subdirectory(test)
