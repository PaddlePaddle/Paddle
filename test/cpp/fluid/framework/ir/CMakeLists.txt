# Legacy IR Pass Tests
cc_test(
  node_test
  SRCS node_test.cc
  DEPS node)

cc_test(
  pass_test
  SRCS pass_test.cc
  DEPS graph pass graph_helper)

cc_test(
  graph_test
  SRCS graph_test.cc
  DEPS graph graph_helper op_registry)

cc_test(
  graph_helper_test
  SRCS graph_helper_test.cc
  DEPS graph graph_helper op_registry)

cc_test(
  graph_to_program_pass_test
  SRCS graph_to_program_pass_test.cc
  DEPS graph_to_program_pass)

cc_test(
  cost_model_test
  SRCS cost_model_test.cc
  DEPS cost_model op_registry)

cc_test(
  test_graph_pattern_detector
  SRCS graph_pattern_detector_test.cc
  DEPS graph_pattern_detector)

cc_test(
  test_op_compat_sensible_pass
  SRCS op_compat_sensible_pass_test.cc
  DEPS op_compat_sensible_pass)

# Fusion pass tests
cc_test(
  test_fc_fuse_pass_cc
  SRCS fc_fuse_pass_test.cc
  DEPS fc_fuse_pass)

cc_test(
  test_fc_lstm_fuse_pass_cc
  SRCS fc_lstm_fuse_pass_test.cc
  DEPS fc_lstm_fuse_pass)

cc_test(
  test_fc_gru_fuse_pass_cc
  SRCS fc_gru_fuse_pass_test.cc
  DEPS fc_gru_fuse_pass)

cc_test(
  test_seqpool_concat_fuse_pass
  SRCS seqpool_concat_fuse_pass_test.cc
  DEPS seqpool_concat_fuse_pass)

cc_test(
  test_seqpool_cvm_concat_fuse_pass
  SRCS seqpool_cvm_concat_fuse_pass_test.cc
  DEPS seqpool_cvm_concat_fuse_pass)

cc_test(
  test_repeated_fc_relu_fuse_pass_cc
  SRCS repeated_fc_relu_fuse_pass_test.cc
  DEPS repeated_fc_relu_fuse_pass)

cc_test(
  test_is_test_pass
  SRCS is_test_pass_test.cc
  DEPS is_test_pass)

cc_test(
  test_simplify_with_basic_ops_pass
  SRCS simplify_with_basic_ops_pass_test.cc
  DEPS simplify_with_basic_ops_pass)

cc_test(
  test_fc_elementwise_layernorm_fuse_pass_cc
  SRCS fc_elementwise_layernorm_fuse_pass_test.cc
  DEPS fc_elementwise_layernorm_fuse_pass)

cc_test(
  test_skip_layernorm_fuse_pass
  SRCS skip_layernorm_fuse_pass_test.cc
  DEPS skip_layernorm_fuse_pass)

cc_test(
  test_multihead_matmul_fuse_pass
  SRCS multihead_matmul_fuse_pass_test.cc
  DEPS multihead_matmul_fuse_pass)

cc_test(
  test_fused_multi_transformer_encoder_pass
  SRCS fused_multi_transformer_encoder_pass_test.cc
  DEPS fused_multi_transformer_encoder_pass)

cc_test(
  test_fused_multi_transformer_decoder_pass
  SRCS fused_multi_transformer_decoder_pass_test.cc
  DEPS fused_multi_transformer_decoder_pass)

cc_test(
  test_fuse_multi_transformer_layer_pass
  SRCS fuse_multi_transformer_layer_pass_test.cc
  DEPS fuse_multi_transformer_layer_pass)

cc_test(
  test_conv_bn_fuse_pass_cc
  SRCS conv_bn_fuse_pass_test.cc
  DEPS conv_bn_fuse_pass)

cc_test(
  test_adaptive_pool2d_convert_global_pass
  SRCS adaptive_pool2d_convert_global_pass_test.cc
  DEPS adaptive_pool2d_convert_global_pass)

cc_test(
  test_generate_pass_cc
  SRCS generate_pass_test.cc
  DEPS generate_pass pass_desc_proto)

# Delete/Cleanup pass tests
cc_test(
  test_delete_op_device_pass
  SRCS delete_op_device_pass_test.cc
  DEPS delete_op_device_pass)

cc_test(
  test_delete_assign_op_pass_cc
  SRCS delete_assign_op_pass_test.cc
  DEPS delete_assign_op_pass)

cc_test(
  test_identity_op_clean_pass_cc
  SRCS identity_op_clean_pass_test.cc
  DEPS identity_op_clean_pass)

cc_test(
  test_delete_dropout_pass_cc
  SRCS delete_dropout_op_pass_test.cc
  DEPS delete_dropout_op_pass)

cc_test(
  test_delete_dequant_weight_linear_op_pass
  SRCS delete_weight_dequant_linear_op_pass_test.cc
  DEPS delete_weight_dequant_linear_op_pass)

cc_test(
  test_delete_cast_op_pass
  SRCS delete_cast_op_pass_test.cc
  DEPS delete_cast_op_pass)

cc_test(
  test_relu6_fuse_pass
  SRCS relu6_fuse_pass_test.cc
  DEPS relu6_fuse_pass)

# GPU/ROCM specific tests
if(WITH_GPU OR WITH_ROCM)
  cc_test(
    test_embedding_eltwise_layernorm_fuse_pass
    SRCS embedding_eltwise_layernorm_fuse_pass_test.cc
    DEPS embedding_eltwise_layernorm_fuse_pass)

  cc_test(
    test_cudnn_placement_pass
    SRCS cudnn_placement_pass_test.cc
    DEPS cudnn_placement_pass)
endif()

# Non-Windows specific tests
if(NOT WIN32)
  cc_test(
    test_sync_batch_norm_pass
    SRCS sync_batch_norm_pass_test.cc
    DEPS sync_batch_norm_pass)

  cc_test(
    test_dense_fc_to_sparse_pass_cc
    SRCS dense_fc_to_sparse_pass_test.cc
    DEPS fc_fuse_pass dense_fc_to_sparse_pass)

  cc_test(
    test_dense_multihead_matmul_to_sparse_pass
    SRCS dense_multihead_matmul_to_sparse_pass_test.cc
    DEPS multihead_matmul_fuse_pass dense_multihead_matmul_to_sparse_pass)
endif()

# OneDNN specific tests
if(WITH_ONEDNN)
  add_subdirectory(onednn)
endif()

# XPU specific tests
if(WITH_XPU)
  add_subdirectory(xpu)
endif()

# fusion_group tests (only on Linux/GPU/ROCM)
if(NOT APPLE
   AND NOT WIN32
   AND (WITH_GPU OR WITH_ROCM))
  add_subdirectory(fusion_group)
endif()
