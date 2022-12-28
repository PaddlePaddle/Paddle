add_subdirectory(manual)
add_subdirectory(generated)

cc_library(
  prim_api
  SRCS all.cc
  DEPS static_utils static_prim_api)
