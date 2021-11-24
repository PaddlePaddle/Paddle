add_subdirectory(utils)

cc_library(eager_api SRCS all.cc DEPS global_utils)
