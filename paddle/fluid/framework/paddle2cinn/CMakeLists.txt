
cc_library(transform_desc SRCS transform_desc.cc DEPS proto_desc cinn_core)

cc_test(transform_desc_test SRCS transform_desc_test.cc DEPS transform_desc)
