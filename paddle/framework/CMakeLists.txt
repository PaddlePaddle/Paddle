# ddim lib
cc_library(ddim SRCS ddim.cc)
cc_test(ddim_test SRCS ddim_test.cc DEPS ddim)
nv_test(dim_test SRCS dim_test.cu DEPS ddim)
cc_test(variable_test SRCS variable_test.cc)
cc_test(scope_test SRCS scope_test.cc)
cc_test(enforce_test SRCS enforce_test.cc)

proto_library(op_desc SRCS op_desc.proto)
cc_library(operator SRCS operator.cc DEPS op_desc protobuf)
cc_test(operator_test SRCS operator_test.cc DEPS operator)
