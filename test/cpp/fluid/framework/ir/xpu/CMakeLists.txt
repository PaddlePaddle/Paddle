# XPU IR Pass Tests

cc_test(
  test_cast_mixed_precision_op_fuse_pass
  SRCS cast_mixed_precision_op_fuse_pass_test.cc
  DEPS cast_mixed_precision_op_fuse_pass)

cc_test(
  test_delete_isolated_node_pass
  SRCS delete_isolated_node_pass_test.cc
  DEPS delete_isolated_node_pass)

cc_test(
  test_fused_multi_transformer_xpu_pass
  SRCS fused_multi_transformer_xpu_pass_test.cc
  DEPS fused_multi_transformer_xpu_pass)

cc_test(
  test_fused_multi_transformer_int8_xpu_quant_pass
  SRCS fused_multi_transformer_int8_xpu_quant_pass_test.cc
  DEPS fused_multi_transformer_int8_xpu_quant_pass)

cc_test(
  test_one_beam_size_fuse_pass
  SRCS one_beam_size_fuse_pass_test.cc
  DEPS one_beam_size_fuse_pass)

cc_test(
  test_stack_fuse_pass
  SRCS stack_fuse_pass_test.cc
  DEPS stack_fuse_pass)

cc_test(
  test_fused_multi_transformer_cachekv_layout_trans_pass
  SRCS fused_multi_transformer_cachekv_layout_trans_pass_test.cc
  DEPS fused_multi_transformer_cachekv_layout_trans_pass)

cc_test(
  test_fused_multi_transformer_int8_cachekv_layout_trans_pass
  SRCS fused_multi_transformer_int8_cachekv_layout_trans_pass_test.cc
  DEPS fused_multi_transformer_int8_cachekv_layout_trans_pass)

cc_test(
  test_multi_encoder_xpu_adaptive_seqlen_fuse_pass
  SRCS multi_encoder_xpu_adaptive_seqlen_fuse_pass_test.cc
  DEPS multi_encoder_xpu_adaptive_seqlen_fuse_pass)

cc_test(
  test_xpu_delete_cast_op_pass
  SRCS xpu_delete_cast_op_pass_test.cc
  DEPS xpu_delete_cast_op_pass)

cc_test(
  test_fold_interp_outsize_fuse_pass
  SRCS fold_interp_outsize_fuse_pass_test.cc
  DEPS fold_interp_outsize_fuse_pass)

cc_test(
  test_fold_two_squeeze2_fuse_pass
  SRCS fold_two_squeeze2_fuse_pass_test.cc
  DEPS fold_two_squeeze2_fuse_pass)

cc_test(
  test_matmul_weight_trans_pass
  SRCS matmul_weight_trans_pass_test.cc
  DEPS matmul_weight_trans_pass)

cc_test(
  test_reshape2_matmul_xpu_fuse_pass
  SRCS reshape2_matmul_xpu_fuse_pass_test.cc
  DEPS reshape2_matmul_xpu_fuse_pass)

cc_test(
  test_fast_where_xpu_fuse_pass
  SRCS fast_where_xpu_fuse_pass_test.cc
  DEPS fast_where_xpu_fuse_pass)

cc_test(
  test_squeeze_excitation_fuse_pass
  SRCS squeeze_excitation_fuse_pass_test.cc
  DEPS squeeze_excitation_fuse_pass)
