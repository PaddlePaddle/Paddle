add_subdirectory(utils)
add_subdirectory(generated)

cc_library(eager_api SRCS all.cc DEPS global_utils eager_scale)
