cc_test(test_mkldnn_op_inplace SRCS mkldnn/test_mkldnn_op_inplace.cc DEPS op_registry softmax_op softmax scope device_context enforce executor)

