cc_library(lite_op_teller SRCS op_teller.cc DEPS lite_full_static framework_proto device_context boost xxhash)
cc_library(lite_engine SRCS engine.cc DEPS lite_full_static framework_proto)
# tensor_utils include tensor.h, which include device_context.h. if not add mkldnn deps, multi-thread make may be fail.
cc_library(lite_tensor_utils SRCS tensor_utils.cc DEPS memcpy lite_full_static framework_proto boost device_context mkldnn)

cc_test(test_lite_engine SRCS test_engine.cc DEPS lite_engine protobuf framework_proto glog gtest analysis)
cc_test(test_lite_tensor_utils SRCS test_tensor_utils.cc DEPS lite_engine lite_tensor_utils)
