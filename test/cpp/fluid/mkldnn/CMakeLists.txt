get_property(paddle_lib GLOBAL PROPERTY PADDLE_LIB_NAME)
cc_test(
  test_mkldnn_op_inplace
  SRCS test_mkldnn_op_inplace.cc
  DEPS executor
       op_registry
       elementwise_add_op
       activation_op
       phi
       scope
       device_context
       enforce
       generated_static_op)
cc_test(
  test_mkldnn_cpu_quantize_pass
  SRCS test_mkldnn_cpu_quantize_pass.cc
  DEPS executor
       op_registry
       activation_op
       conv_activation_mkldnn_fuse_pass
       cpu_quantize_placement_pass
       cpu_quantize_pass
       phi
       scope
       device_context)

cc_test(
  test_conv_mkldnn_nhwc
  SRCS test_conv_mkldnn_nhwc.cc
  DEPS executor
       op_registry
       depthwise_conv
       tensor
       phi
       scope
       device_context
       enforce
       generated_static_op)

set(TEST_MKLDNN_CACHING_DEPS
    op_registry
    elementwise_mul_op
    elementwise_add_op
    activation_op
    phi
    scope
    device_context
    enforce
    generated_static_op)

if(WITH_GPU OR WITH_ROCM)
  set(TEST_MKLDNN_CACHING_DEPS ${TEST_MKLDNN_CACHING_DEPS} depthwise_conv)
endif()
cc_test(
  test_mkldnn_caching
  SRCS test_mkldnn_caching.cc
  DEPS ${TEST_MKLDNN_CACHING_DEPS})
if(WIN32 AND WITH_TESTING)
  cc_test_old(
    test_mkldnn_op_nhwc
    SRCS
    test_mkldnn_op_nhwc.cc
    DEPS
    fleet_executor
    conditional_block_op
    standalone_executor
    executor
    recurrent_op_helper
    recurrent_op
    op_registry
    generated_static_op
    crop_op
    activation_op
    generated_op
    generated_static_op
    phi
    transpose_op
    fused_transpose_op
    scope
    device_context
    enforce)
else()
  cc_test_old(test_mkldnn_op_nhwc SRCS test_mkldnn_op_nhwc.cc DEPS
              ${paddle_lib} python)
endif()
