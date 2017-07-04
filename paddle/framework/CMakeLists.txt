# ddim lib
cc_library(ddim SRCS ddim.cc)
cc_test(ddim_test SRCS ddim_test.cc DEPS ddim)
nv_test(dim_test SRCS dim_test.cu DEPS ddim)
cc_test(variable_test SRCS variable_test.cc)
cc_test(scope_test SRCS scope_test.cc)
cc_test(enforce_test SRCS enforce_test.cc)
proto_library(attr_type SRCS attr_type.proto)
proto_library(op_proto SRCS op_proto.proto)

cc_test(op_proto_test SRCS op_proto_test.cc DEPS op_proto attr_type protobuf)

proto_library(net_proto SRCS net_proto.proto)
cc_library(net SRCS net.cc DEPS net_proto attr_type op_proto)
