set(TEST_MKLDNN_CACHING_DEPS op_registry elementwise_mul_op elementwise_add_op activation_op softmax_op conv_op im2col vol2col softmax scope device_context enforce)
if (WITH_GPU OR WITH_ROCM)
  set(TEST_MKLDNN_CACHING_DEPS ${TEST_MKLDNN_CACHING_DEPS} depthwise_conv)
endif()
cc_test(test_mkldnn_caching SRCS mkldnn/test_mkldnn_caching.cc DEPS ${TEST_MKLDNN_CACHING_DEPS})

