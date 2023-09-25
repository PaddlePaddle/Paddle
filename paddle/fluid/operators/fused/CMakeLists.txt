include(operators)
if(WITH_UNITY_BUILD)
  # Load Unity Build rules for operators in paddle/fluid/operators/fused.
  include(unity_build_rule.cmake)
endif()
register_operators(
  EXCLUDES
  fused_bn_activation_op
  conv_fusion_op
  fusion_transpose_flatten_concat_op
  fusion_conv_inception_op
  fused_fc_elementwise_layernorm_op
  self_dp_attention_op
  skip_layernorm_op
  yolo_box_head_op
  yolo_box_post_op
  fused_embedding_eltwise_layernorm_op
  fusion_group_op
  fusion_gru_op
  fusion_lstm_op
  fused_bn_add_activation_op
  fused_attention_op
  fused_transformer_op
  fused_feedforward_op
  fused_multi_transformer_op
  fused_multi_transformer_int8_op
  fused_bias_dropout_residual_layer_norm_op
  resnet_unit_op
  fused_gemm_epilogue_op
  fused_gate_attention_op
  resnet_basic_block_op)

# fusion_gru_op does not have CUDA kernel
op_library(fusion_gru_op)
op_library(fusion_lstm_op)
if(WITH_AVX
   AND AVX512F_FOUND
   AND AVX512F_FLAG
   AND WITH_MKL)
  op_library(self_dp_attention_op)
  set_target_properties(
    self_dp_attention_op
    PROPERTIES COMPILE_FLAGS "-Wno-maybe-uninitialized -mfma ${AVX512F_FLAG}")
endif()

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
  # conv_fusion_op needs cudnn 7 above
  if(NOT ${CUDNN_VERSION} VERSION_LESS 7100)
    op_library(conv_fusion_op)
  endif()
  # fusion_transpose_flatten_concat_op
  # HIP not support cudnnTransformTensor
  if(NOT WITH_ROCM)
    op_library(fusion_transpose_flatten_concat_op)
  endif()
  # fusion_conv_inception_op needs cudnn 7 above
  # HIP not support cudnnConvolutionBiasActivationForward
  if((NOT WITH_ROCM) AND (NOT ${CUDNN_VERSION} VERSION_LESS 7100))
    op_library(fusion_conv_inception_op)
  endif()
  # fused_fc_elementwise_layernorm_op
  op_library(fused_fc_elementwise_layernorm_op)
  op_library(skip_layernorm_op)
  op_library(yolo_box_head_op)
  op_library(yolo_box_post_op)
  op_library(fused_embedding_eltwise_layernorm_op DEPS bert_encoder_functor)
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
    op_library(fused_bias_dropout_residual_layer_norm_op)
  endif()
  # resnet_unit needs cudnn 8.0 above
  if((NOT WITH_ROCM) AND (NOT ${CUDNN_VERSION} VERSION_LESS 8000))
    op_library(resnet_unit_op)
  endif()

  if(CUDA_VERSION GREATER_EQUAL 11.6)
    op_library(fused_gemm_epilogue_op)
  endif()
endif()
