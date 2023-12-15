include(operators)
if(WITH_UNITY_BUILD)
  # Load Unity Build rules for operators in paddle/fluid/operators/fused.
  include(unity_build_rule.cmake)
endif()
register_operators(
  EXCLUDES
  fused_bn_activation_op
  fusion_conv_inception_op
  yolo_box_head_op
  yolo_box_post_op
  fusion_group_op
  fusion_lstm_op
  fused_bn_add_activation_op
  fused_attention_op
  fused_transformer_op
  fused_feedforward_op
  fused_multi_transformer_op
  fused_multi_transformer_int8_op
  resnet_unit_op
  fused_gemm_epilogue_op
  fused_gate_attention_op
  resnet_basic_block_op)

op_library(fusion_lstm_op)

if(WITH_XPU)
  op_library(resnet_basic_block_op)
  op_library(resnet_unit_op)
  op_library(fused_gemm_epilogue_op)
  op_library(fused_attention_op)
  op_library(fused_feedforward_op)
endif()

if(WITH_GPU OR WITH_ROCM)
  # fused_bn_activation_op needs cudnn 7.4.1 above
  # HIP not support bn act fuse in MIOPEN
  if((NOT WITH_ROCM) AND (NOT ${CUDNN_VERSION} VERSION_LESS 7401))
    op_library(fused_bn_activation_op)
  endif()
  # HIP not support cudnnTransformTensor
  # fusion_conv_inception_op needs cudnn 7 above
  # HIP not support cudnnConvolutionBiasActivationForward
  if((NOT WITH_ROCM) AND (NOT ${CUDNN_VERSION} VERSION_LESS 7100))
    op_library(fusion_conv_inception_op)
  endif()
  op_library(yolo_box_head_op)
  op_library(yolo_box_post_op)
  op_library(fused_gate_attention_op)
  # fusion_group
  if(NOT APPLE AND NOT WIN32)
    op_library(fusion_group_op)
  endif()
  # fused_bn_add_activation
  # HIP not support bn act fuse in MIOPEN
  if((NOT WITH_ROCM) AND (NOT ${CUDNN_VERSION} VERSION_LESS 7401))
    op_library(fused_bn_add_activation_op)
  endif()
  # fused_dropout
  # only support CUDA
  if(NOT WITH_ROCM)
    op_library(fused_feedforward_op)
    # fused_attention_op
    op_library(fused_attention_op)
    op_library(fused_multi_transformer_op)
    op_library(fused_multi_transformer_int8_op)
  endif()
  # resnet_unit needs cudnn 8.0 above
  if((NOT WITH_ROCM) AND (NOT ${CUDNN_VERSION} VERSION_LESS 8000))
    op_library(resnet_unit_op)
  endif()

  if(CUDA_VERSION GREATER_EQUAL 11.6)
    op_library(fused_gemm_epilogue_op)
  endif()
endif()
