add_subdirectory(cpu)
cc_library(pten_context SRCS all_context.cc DEPS device_context)
