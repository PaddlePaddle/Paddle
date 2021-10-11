add_subdirectory(tests)

cc_library(exp_lib SRCS allocator.cc storage.cc tensor.cc DEPS tcmpt_core)
