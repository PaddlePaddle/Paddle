set(PATTERN_REWRITE_TEST_DEPS
    pd_constant_folding_pass transform_general_functions gtest pd_op_dialect
    pir)

if(WITH_DISTRIBUTE)
  set(PATTERN_REWRITE_TEST_DEPS ${PATTERN_REWRITE_TEST_DEPS} fleet_executor
                                parallel_executor)
endif()

cc_test_old(pattern_rewrite_test SRCS pattern_rewrite_test.cc DEPS
            ${PATTERN_REWRITE_TEST_DEPS})

cc_test_old(
  drr_test
  SRCS
  drr_test.cc
  DEPS
  drr
  gtest
  pd_op_dialect
  pir)
cc_test_old(
  drr_same_type_binding_test
  SRCS
  drr_same_type_binding_test.cc
  DEPS
  drr
  gtest
  pd_op_dialect
  pir)

cc_test_old(
  drr_fuse_linear_test
  SRCS
  drr_fuse_linear_test.cc
  DEPS
  fusion_passes
  drr
  gtest
  pd_op_dialect
  pir)
cc_test_old(
  drr_attention_fuse_test
  SRCS
  drr_attention_fuse_test.cc
  DEPS
  fusion_passes
  drr
  gtest
  pd_op_dialect
  pir)

set_tests_properties(
  pattern_rewrite_test PROPERTIES ENVIRONMENT
                                  "FLAGS_enable_new_ir_in_executor=true")
