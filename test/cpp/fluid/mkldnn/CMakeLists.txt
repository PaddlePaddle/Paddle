get_property(paddle_lib GLOBAL PROPERTY PADDLE_LIB_NAME)
paddle_test(test_mkldnn_op_inplace SRCS test_mkldnn_op_inplace.cc)
paddle_test(test_mkldnn_cpu_quantize_pass SRCS test_mkldnn_cpu_quantize_pass.cc)

paddle_test(test_conv_mkldnn_nhwc SRCS test_conv_mkldnn_nhwc.cc)

set(TEST_MKLDNN_CACHING_DEPS
    op_registry
    elementwise_mul_op
    elementwise_add_op
    activation_op
    phi
    common
    scope
    device_context
    enforce
    generated_static_op)

if(WITH_GPU OR WITH_ROCM)
  set(TEST_MKLDNN_CACHING_DEPS ${TEST_MKLDNN_CACHING_DEPS} depthwise_conv)
endif()
paddle_test(test_mkldnn_caching SRCS test_mkldnn_caching.cc)

if(WITH_TESTING)
  paddle_test(test_mkldnn_op_nhwc SRCS test_mkldnn_op_nhwc.cc)
endif()

paddle_test(test_mkldnn_pool_adaptive_op SRCS test_mkldnn_pool_adaptive_op.cc)

paddle_test(test_mkldnn_squeeze SRCS test_mkldnn_squeeze.cc)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(test_mkldnn_op_nhwc)
endif()
