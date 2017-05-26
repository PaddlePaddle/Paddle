# gangliao's built in cmake function

file(GLOB_RECURSE OPTIMIZER_HEADER RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.h")
file(GLOB_RECURSE OPTIMIZER_CC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cc")

cc_library(optimizer SRCS
  OPTIMIZER_CC)

cc_test(optimizer_test
  SRCS optimizer_test.cc
  DEPS optimizer)
