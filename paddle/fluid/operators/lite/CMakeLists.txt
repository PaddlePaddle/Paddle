op_library(lite_engine_op DEPS lite_engine lite_tensor_utils)
cc_test(test_lite_engine_op SRCS lite_engine_op_test.cc DEPS lite_engine_op analysis)
