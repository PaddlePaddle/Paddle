add_subdirectory(general_fusion_merge_pass)

core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  infershape.cc
  opfusion.cc
  alterlayout.cc
  const_propagate.cc
  op_fusion_pass.cc
  fusion_merge_pass.cc
  general_fusion_merge_pass.cc
  dot_merger.cc
  check_fusion_accuracy_pass.cc
  custom_call_pass.cc
  common_subexpression_elimination.cc
  constant_folding_pass.cc
  dce_pass.cc
  dense_merge_pass.cc
  reduce_split_pass.cc
  single_group_optimize_pass.cc
  constant_folding_pass_util.cc)

#cinn_cc_test(test_opfusion SRCS opfusion_test.cc DEPS cinncore)
if(WITH_CUDA)
  cinn_cc_test(test_primitive_ops SRCS test_primitive_ops.cc DEPS cinncore)
  cinn_cc_test(test_const_propagate SRCS const_propagate_test.cc DEPS cinncore)
  cinn_cc_test(test_dot_merger_pass SRCS dot_merger_test.cc DEPS cinncore)

  # TODO(thisjiang): move when test bug in x86 is fixed
  cinn_cc_test(
    test_check_fusion_accuracy_pass SRCS check_fusion_accuracy_pass_test.cc
    DEPS cinncore decomposer_test_helper)
  cinn_cc_test(test_dense_merge_pass SRCS dense_merge_pass_test.cc DEPS
               cinncore)
  cinn_cc_test(test_reduce_split_pass SRCS reduce_split_pass_test.cc DEPS
               cinncore)
endif()
cinn_cc_test(test_op_fusion_pass SRCS op_fusion_pass_test.cc DEPS cinncore
             decomposer_test_helper)
cinn_cc_test(test_fusion_merge_pass SRCS fusion_merge_pass_test.cc DEPS
             cinncore decomposer_test_helper)
if(NOT WITH_CUDA)
  #cinn_cc_test(test_alterlayout SRCS alterlayout_test.cc DEPS cinncore)
endif()
cinn_cc_test(test_dot_merger SRCS test_dot_merger.cc DEPS cinncore)
cinn_cc_test(test_dce_pass SRCS dce_pass_test.cc DEPS cinncore)
cinn_cc_test(test_common_subexpression_elimination SRCS
             common_subexpression_elimination_test.cc DEPS cinncore)
cinn_cc_test(test_constant_folding_pass SRCS constant_folding_pass_test.cc DEPS
             cinncore)
