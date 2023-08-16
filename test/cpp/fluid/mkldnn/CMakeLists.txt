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
pass_library(simplify_with_basic_ops_pass base)
pass_library(quant_dequant_mkldnn_pass inference DIR mkldnn)
pass_library(conv_bn_fuse_pass inference)
pass_library(conv_affine_channel_mkldnn_fuse_pass inference DIR mkldnn)
pass_library(conv_bias_mkldnn_fuse_pass inference DIR mkldnn)
pass_library(conv_elementwise_add_mkldnn_fuse_pass inference DIR mkldnn)
pass_library(conv_activation_mkldnn_fuse_pass inference DIR mkldnn)
pass_library(operator_scale_onednn_fuse_pass inference DIR mkldnn)
pass_library(operator_unsqueeze2_onednn_fuse_pass inference DIR mkldnn)
pass_library(operator_reshape2_onednn_fuse_pass inference DIR mkldnn)
pass_library(cpu_quantize_placement_pass base DIR mkldnn)
pass_library(cpu_quantize_pass inference DIR mkldnn)
cc_test(
  test_mkldnn_cpu_quantize_pass
  SRCS test_mkldnn_cpu_quantize_pass.cc
  DEPS executor
       op_registry
       activation_op
       simplify_with_basic_ops_pass
       quant_dequant_mkldnn_pass
       conv_bn_fuse_pass
       conv_affine_channel_mkldnn_fuse_pass
       conv_bias_mkldnn_fuse_pass
       conv_elementwise_add_mkldnn_fuse_pass
       conv_activation_mkldnn_fuse_pass
       operator_scale_onednn_fuse_pass
       operator_unsqueeze2_onednn_fuse_pass
       operator_reshape2_onednn_fuse_pass
       cpu_quantize_placement_pass
       cpu_quantize_pass
       phi
       scope
       device_context)

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
