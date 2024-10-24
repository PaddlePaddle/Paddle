list(
  APPEND
  TRT_FILES
  trt_plugin.cc
  split_op_plugin.cu
  elementwise_op_plugin.cu
  gelu_op_plugin.cu
  pool_op_plugin.cu
  swish_op_plugin.cu
  group_norm_op_plugin.cu
  layer_norm_op_plugin.cu
  instance_norm_op_plugin.cu
  qkv_to_context_plugin.cu
  hard_swish_op_plugin.cu
  stack_op_plugin.cu
  anchor_generator_op_plugin.cu
  yolo_box_op_plugin.cu
  yolo_box_head_op_plugin.cu
  roi_align_op_plugin.cu
  gather_nd_op_plugin.cu
  mish_op_plugin.cu
  pool3d_op_plugin.cu
  deformable_conv_op_plugin.cu
  matmul_op_int8_plugin.cu
  multihead_matmul_roformer_plugin.cu
  transformer_input_output_convert_plugin.cu
  remove_padding_plugin.cu
  recover_padding_plugin.cu
  c_allreduce_op_plugin.cu
  preln_residual_bias_plugin.cu
  fused_token_prune_op_plugin.cu
  layernorm_shift_partition_op.cu
  reverse_roll_op_plugin.cu
  prelnlayernorm_shift_partition_op.cu
  trans_layernorm_op_plugin.cu
  merge_layernorm_op_plugin.cu
  skip_merge_layernorm_op_plugin.cu
  skip_groupnorm_act_op_plugin.cu
  preln_groupnorm_act_op_plugin.cu
  elementwiseadd_transpose_op_plugin.cu
  generic_plugin.cu
  custom_generic_plugin.cu
  many_emb_layernorm_plugin.cu
  many_emb_layernorm_kernel.cu
  prompt_tuning_emb_layernorm_varseqlen_kernel_hface.cu
  prompt_tuning_emb_layernorm_varseqlen_plugin.cu)

if(${TENSORRT_MAJOR_VERSION} GREATER_EQUAL 7)
  list(APPEND TRT_FILES many_emb_layernorm_varseqlen_plugin.cu
       many_emb_layernorm_varseqlen_kernel_mtron.cu
       many_emb_layernorm_varseqlen_kernel_hface.cu)
endif()

if(CUSPARSELT_FOUND AND ${TENSORRT_MAJOR_VERSION} GREATER_EQUAL 8)
  list(APPEND TRT_FILES spmm_plugin.cu)
endif()

nv_library(
  tensorrt_plugin
  SRCS ${TRT_FILES}
  DEPS phi tensorrt_engine tensor common
       tensorrt_dynamic_shape_infermeta_factory
       tensorrt_plugin_arg_mapping_context)

nv_test(
  test_split_plugin
  SRCS test_split_plugin.cc
  DEPS paddle_framework ${GLOB_OPERATOR_DEPS} tensorrt_plugin)

if(NOT WIN32)
  nv_test(
    test_fused_token_prune_plugin
    SRCS test_fused_token_prune_plugin.cc
    DEPS paddle_framework ${GLOB_OPERATOR_DEPS} tensorrt_plugin)
endif()
