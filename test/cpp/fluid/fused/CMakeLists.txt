if(WITH_GPU OR WITH_ROCM)
  # fusion_group
  if(NOT APPLE AND NOT WIN32)
    cc_test(
      test_fusion_group_op
      SRCS fusion_group_op_test.cc
      DEPS fusion_group_op)
  endif()
  if(NOT WITH_ROCM)
    nv_test(
      test_fused_residual_dropout_bias
      SRCS fused_residual_dropout_bias_test.cu
      DEPS tensor
           op_registry
           dropout_op
           generated_op
           device_context
           phi
           common
           memory)
    nv_test(
      test_fused_dropout_act_bias
      SRCS fused_dropout_act_bias_test.cu
      DEPS tensor
           op_registry
           dropout_op
           generated_op
           device_context
           phi
           common
           memory)
    nv_test(
      test_fused_layernorm_residual_dropout_bias
      SRCS fused_layernorm_residual_dropout_bias_test.cu
      DEPS fleet_executor
           tensor
           op_registry
           dropout_op
           generated_op
           device_context
           phi
           common
           memory
           ${CINN_DEPS})
  endif()
  # resnet_unit needs cudnn 8.0 above
  if((NOT WITH_ROCM) AND (NOT ${CUDNN_VERSION} VERSION_LESS 8000))
    cc_test(
      test_cudnn_norm_conv
      SRCS cudnn_norm_conv_test.cc
      DEPS generated_op
           depthwise_conv
           tensor
           op_registry
           device_context
           phi
           common
           memory)
    cc_test(
      test_cudnn_bn_add_relu
      SRCS cudnn_bn_add_relu_test.cc
      DEPS batch_norm_op
           fused_bn_add_activation_op
           tensor
           op_registry
           device_context
           phi
           common
           memory)
  endif()
endif()
