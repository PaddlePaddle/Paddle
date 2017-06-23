cc_library(ddim SRCS ddim.cc)
cc_test(ddim_test SRCS ddim_test.cc DEPS ddim)

nv_test(dim_test SRCS dim_test.cu DEPS ddim)

proto_library(example_proto SRCS example.proto)
cc_binary(example_proto_exe SRCS example.cc DEPS example_proto protobuf)
