if(WITH_GPU OR WITH_ROCM)
  # fusion_group
  if(NOT APPLE AND NOT WIN32)
    cc_test(
      test_fusion_group_op
      SRCS fusion_group_op_test.cc
      DEPS fusion_group_op)
  endif()
  # resnet_unit needs cudnn 8.0 above
  if((NOT WITH_ROCM) AND (NOT ${CUDNN_VERSION} VERSION_LESS 8000))
    cc_test(
      test_cudnn_norm_conv
      SRCS cudnn_norm_conv_test.cc
      DEPS conv_op
           blas
           im2col
           vol2col
           depthwise_conv
           eigen_function
           tensor
           op_registry
           device_context
           generator
           memory)
    cc_test(
      test_cudnn_bn_add_relu
      SRCS cudnn_bn_add_relu_test.cc
      DEPS batch_norm_op
           fused_bn_add_activation_op
           tensor
           op_registry
           device_context
           generator
           memory)
  endif()
endif()
