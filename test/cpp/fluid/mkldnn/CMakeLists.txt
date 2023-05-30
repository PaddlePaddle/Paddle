cc_test(
  test_mkldnn_op_inplace
  SRCS test_mkldnn_op_inplace.cc
  DEPS op_registry
       elementwise_add_op
       activation_op
       phi
       scope
       device_context
       enforce
       executor
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

cc_test_old(
  test_mkldnn_op_nhwc
  SRCS
  test_mkldnn_op_nhwc.cc
  DEPS
  op_registry
  pool_op
  shape_op
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
  executor)
