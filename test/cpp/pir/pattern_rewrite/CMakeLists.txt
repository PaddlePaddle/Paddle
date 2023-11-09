set(PATTERN_REWRITE_TEST_DEPS transform gtest op_dialect_vjp pir)

if(WITH_DISTRIBUTE)
  set(PATTERN_REWRITE_TEST_DEPS ${PATTERN_REWRITE_TEST_DEPS} fleet_executor
                                parallel_executor)
endif()

cc_test_old(pattern_rewrite_test SRCS pattern_rewrite_test.cc DEPS
            ${PATTERN_REWRITE_TEST_DEPS})

cc_test(
  drr_test
  SRCS drr_test.cc
  DEPS drr transform)

cc_test(
  drr_same_type_binding_test
  SRCS drr_same_type_binding_test.cc
  DEPS drr gtest op_dialect_vjp pir transform)

cc_test(
  drr_fuse_linear_test
  SRCS drr_fuse_linear_test.cc
  DEPS transform drr gtest op_dialect_vjp pir)

cc_test(
  drr_fuse_linear_param_grad_add_test
  SRCS drr_fuse_linear_param_grad_add_test.cc
  DEPS transform drr gtest op_dialect_vjp pir)

cc_test(
  drr_attention_fuse_test
  SRCS drr_attention_fuse_test.cc
  DEPS transform drr gtest op_dialect_vjp pir)

set_tests_properties(pattern_rewrite_test
                     PROPERTIES ENVIRONMENT "FLAGS_enable_pir_in_executor=true")
