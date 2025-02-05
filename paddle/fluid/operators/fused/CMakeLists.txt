include(operators)
if(WITH_UNITY_BUILD)
  # Load Unity Build rules for operators in paddle/fluid/operators/fused.
  include(unity_build_rule.cmake)
endif()
register_operators(
  EXCLUDES
  fused_bn_activation_op
  fused_bn_add_activation_op
  fused_attention_op
  fused_transformer_op
  fused_feedforward_op
  fused_multi_transformer_int8_op
  fused_gemm_epilogue_op
  fused_gate_attention_op)

if(WITH_XPU)
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
  # HIP not support cudnnConvolutionBiasActivationForward
  op_library(fused_gate_attention_op)
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
    op_library(fused_multi_transformer_int8_op)
  endif()

  if(CUDA_VERSION GREATER_EQUAL 11.6)
    op_library(fused_gemm_epilogue_op)
  endif()
  if(WITH_ROCM)
    op_library(fused_gemm_epilogue_op)
  endif()
endif()
