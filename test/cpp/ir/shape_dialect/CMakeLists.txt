cc_test_old(
  symbolic_op_test
  SRCS
  symbolic_op_test.cc
  DEPS
  pd_dialect
  ir
  gtest)

set(TEST_DEPS gtest pd_dialect ir)

if(WITH_DISTRIBUTE)
  set(TEST_DEPS ${TEST_DEPS} fleet_executor)
endif()

cc_test_old(constraint_pass_test SRCS constraint_pass_test.cc DEPS ${TEST_DEPS})

set_tests_properties(
  constraint_pass_test PROPERTIES ENVIRONMENT
                                  "FLAGS_enable_new_ir_in_executor=true")
