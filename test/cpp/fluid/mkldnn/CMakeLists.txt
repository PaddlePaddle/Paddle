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
       
set(CPU_QUANTIZE_PASS_TEST_APP "test_analyzer_cpu_quantize_pass")
set(CPU_QUANTIZE_PASS_TEST_APP_SRC
    "test_mkldnn_cpu_quantize_pass.cc")
    inference_analysis_api_test_build(${CPU_QUANTIZE_PASS_TEST_APP}
                                      ${CPU_QUANTIZE_PASS_TEST_APP_SRC})

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
