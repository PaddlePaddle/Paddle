# Add TRT tests
list(
  APPEND
  CONVERT_FILES
  matmul_op.cc
  conv2d_op.cc
  fc_op.cc
  pool2d_op.cc
  elementwise_op.cc
  batch_norm_op.cc
  activation_op.cc
  unary_op.cc
  softmax_op.cc
  concat_op.cc
  dropout_op.cc
  group_norm_op.cc
  pad_op.cc
  split_op.cc
  prelu_op.cc
  leaky_relu_op.cc
  gelu_op.cc
  layer_norm_op.cc
  multihead_matmul_op.cc
  shuffle_channel_op.cc
  swish_op.cc
  silu_op.cc
  instance_norm_op.cc
  stack_op.cc
  transpose_op.cc
  flatten_op.cc
  flatten_contiguous_range_op.cc
  emb_eltwise_layernorm.cc
  skip_layernorm.cc
  scale_op.cc
  slice_op.cc
  hard_sigmoid_op.cc
  hard_swish_op.cc
  clip_op.cc
  gather_op.cc
  anchor_generator_op.cc
  yolo_box_op.cc
  yolo_box_head_op.cc
  arg_max_op.cc
  roi_align_op.cc
  affine_channel_op.cc
  multiclass_nms_op.cc
  multiclass_nms3_op.cc
  nearest_interp_op.cc
  reshape_op.cc
  reduce_op.cc
  gather_nd_op.cc
  tile_op.cc
  conv3d_op.cc
  mish_op.cc
  nearest_interp_v2_op.cc
  bilinear_interp_v2_op.cc
  pool3d_op.cc
  deformable_conv_op.cc
  preln_emb_eltwise_layernorm.cc
  strided_slice_op.cc
  preln_skip_layernorm.cc
  roll_op.cc
  transformer_input_convert_op.cc
  cast_op.cc
  remove_padding_op.cc
  equal_op.cc
  recover_padding_op.cc
  preln_residual_bias.cc
  c_allreduce_op.cc
  top_k_op.cc
  squeeze2_op.cc
  unsqueeze2_op.cc
  rnn_op.cc
  fill_constant_batch_size_like_op.cc
  sum_op.cc
  shape_op.cc
  fill_constant_op.cc
  fused_token_prune_op.cc
  layernorm_shift_partition_op.cc
  merge_layernorm_op.cc
  generic_and_custom_plugin_creater.cc)

if(CUSPARSELT_FOUND AND ${TENSORRT_MAJOR_VERSION} GREATER_EQUAL 8)
  list(APPEND CONVERT_FILES sparse_fc_op.cc sparse_multihead_matmul_op.cc)
endif()

nv_library(
  tensorrt_converter
  SRCS ${CONVERT_FILES}
  DEPS tensorrt_engine
       tensorrt_plugin
       operator
       scope
       framework_proto
       tensorrt_op_teller
       op_registry)

nv_test(
  test_op_converter
  SRCS test_op_converter.cc
  DEPS paddle_framework ${GLOB_OPERATOR_DEPS} tensorrt_engine
       tensorrt_converter)

nv_test(
  test_custom_plugin_creater
  SRCS test_custom_plugin_creater.cc
  DEPS paddle_framework
       ${GLOB_OPERATOR_DEPS}
       tensorrt_engine
       tensorrt_plugin
       tensorrt_converter
       op_meta_info
       custom_operator)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(test_op_converter)
endif()
