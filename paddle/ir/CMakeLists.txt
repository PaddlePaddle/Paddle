if(NOT WITH_NEWIR)
  return()
endif()

add_subdirectory(type)

cc_library(
  ir_context
  SRCS ir_context.cc
  DEPS storage_uniquer)
