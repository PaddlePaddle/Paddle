cc_library(ddim SRCS ddim.cc)
cc_test(ddim_test SRCS ddim_test.cc DEPS ddim)

nv_test(dim_test SRCS dim_test.cu DEPS ddim)
cc_test(typedefs_test SRCS typedefs_test.cc)
