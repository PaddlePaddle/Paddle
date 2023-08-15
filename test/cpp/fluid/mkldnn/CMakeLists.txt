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
       conv2d_op
       activation_op
       cpu_quantize_pass
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
if(WITH_CINN)
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
    cinn_compiler
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
    enforce
    python)
else()
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
endif()
