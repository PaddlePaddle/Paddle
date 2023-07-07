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

if(WITH_CUDA)
  cinn_cc_test(test_decomposer_pass SRCS decomposer_test.cc DEPS cinncore)
  cinn_cc_test(test_dead_code_eliminate_pass SRCS dead_code_eliminate_test.cc
               DEPS cinncore)
  cinn_cc_test(test_remove_identity_pass SRCS remove_identity_test.cc DEPS
               cinncore)
  cinn_cc_test(test_fill_constant_rewriter_pass SRCS
               fill_constant_rewriter_test.cc DEPS cinncore)
  cinn_cc_test(test_fill_constant_folding_pass SRCS
               fill_constant_folding_test.cc DEPS cinncore)
  cinn_cc_test(test_program_topoerror SRCS program_topoerror_test.cc DEPS
               cinncore)
endif()
if(WITH_CUDNN)
  cinn_cc_test(test_gemm_rewriter_pass SRCS gemm_rewriter_test.cc DEPS cinncore)
  cinn_cc_test(test_transpose_folding_input_pass SRCS
               transpose_folding_input_test.cc DEPS cinncore)
  cinn_cc_test(test_transpose_folding_output_pass SRCS
               transpose_folding_output_test.cc DEPS cinncore)
  cinn_cc_test(test_transpose_scale_folding SRCS
               transpose_scale_folding_test.cc DEPS cinncore)
endif()
cinn_cc_test(test_transpose_collapsing SRCS transpose_collapsing_test.cc DEPS
             cinncore)
cinn_cc_test(test_cast_collapsing SRCS cast_collapsing_test.cc DEPS cinncore)
cinn_cc_test(test_auto_cast SRCS auto_cast_test.cc DEPS cinncore)
cinn_cc_test(test_expand_zero_dim_pass SRCS expand_zero_dim_pass_test.cc DEPS
             cinncore)
