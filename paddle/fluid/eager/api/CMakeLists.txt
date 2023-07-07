add_subdirectory(manual)
add_subdirectory(utils)
add_subdirectory(generated)

cc_library(
  eager_api
  SRCS all.cc
  DEPS hook_utils global_utils eager_scale)
