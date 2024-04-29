add_subdirectory(eager)
add_subdirectory(static)
cc_library(
  prim_utils
  SRCS utils.cc
  DEPS static_global_utils)
