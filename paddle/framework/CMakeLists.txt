cc_library(ddim SRCS ddim.cc)
cc_test(ddim_test SRCS ddim_test.cc DEPS ddim)
nv_test(dim_test SRCS dim_test.cu DEPS ddim)

cc_test(variable_test SRCS variable_test.cc)
# include generated protobuf headers
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
generate_protobuf_cpp(attr_proto_src attr_proto_hdr attr.proto)
cc_library(attr_proto SRCS ${attr_proto_src})
generate_protobuf_cpp(attr_test_proto_src attr_test_proto_header attr_test.proto)
message(STATUS ${attr_test_proto_src})
cc_test(attr_test SRCS ${attr_test_proto_src} attr_test.cc
        DEPS attr_proto protobuf glog gflags)
