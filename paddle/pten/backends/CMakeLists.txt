add_subdirectory(dynload)

cc_library(pten_context SRCS all_context.cc DEPS device_context)
