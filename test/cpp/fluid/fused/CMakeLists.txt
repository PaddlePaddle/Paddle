if(WITH_GPU OR WITH_ROCM)
  # fusion_group
  if(NOT APPLE AND NOT WIN32)
    paddle_test(test_fusion_group_op SRCS fusion_group_op_test.cc)
  endif()
  if(NOT WITH_ROCM)
    paddle_test(test_fused_residual_dropout_bias SRCS
                fused_residual_dropout_bias_test.cu)
    paddle_test(test_fused_dropout_act_bias SRCS fused_dropout_act_bias_test.cu)
    paddle_test(test_fused_layernorm_residual_dropout_bias SRCS
                fused_layernorm_residual_dropout_bias_test.cu)
  endif()
  # resnet_unit needs cudnn 8.0 above
  if((NOT WITH_ROCM) AND (NOT ${CUDNN_VERSION} VERSION_LESS 8000))
    paddle_test(test_cudnn_norm_conv SRCS cudnn_norm_conv_test.cc DEPS
                generated_op)
    paddle_test(test_cudnn_bn_add_relu SRCS cudnn_bn_add_relu_test.cc)
  endif()
endif()
