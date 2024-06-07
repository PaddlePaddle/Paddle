core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  decomposer.cc
  dead_code_eliminate.cc
  remove_identity.cc
  transpose_collapsing.cc
  transpose_folding_input.cc
  transpose_folding_output.cc
  gemm_rewriter.cc
  fill_constant_rewriter.cc
  fill_constant_folding.cc
  cast_collapsing.cc
  auto_cast.cc
  expand_zero_dim_pass.cc
  auto_broadcast.cc)

cinn_cc_test(test_auto_cast SRCS auto_cast_test.cc DEPS cinncore)
