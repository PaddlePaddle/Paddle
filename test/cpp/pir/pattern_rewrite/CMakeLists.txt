set(PATTERN_REWRITE_TEST_DEPS
    _constant_folding_pass transform_general_functions gtest pd_op_dialect pir)

if(WITH_DISTRIBUTE)
  set(PATTERN_REWRITE_TEST_DEPS ${PATTERN_REWRITE_TEST_DEPS} fleet_executor
                                parallel_executor)
endif()

cc_test_old(pattern_rewrite_test SRCS pattern_rewrite_test.cc DEPS
            ${PATTERN_REWRITE_TEST_DEPS})

set_tests_properties(
  pattern_rewrite_test PROPERTIES ENVIRONMENT
                                  "FLAGS_enable_new_ir_in_executor=true")
