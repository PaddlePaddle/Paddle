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

if(NOT WITH_CUDA)
  #cinn_cc_test(test_alterlayout SRCS alterlayout_test.cc DEPS cinncore)
endif()
# cinn_cc_test(test_dce_pass SRCS dce_pass_test.cc DEPS cinncore)
