cc_test_old(
  symbolic_op_test
  SRCS
  symbolic_op_test.cc
  DEPS
  pd_op_dialect
  pir
  gtest)

cc_test_old(
  constraint_pass_test
  SRCS
  constraint_pass_test.cc
  DEPS
  gtest
  pd_op_dialect
  pir)

set_tests_properties(
  constraint_pass_test PROPERTIES ENVIRONMENT
                                  "FLAGS_enable_new_ir_in_executor=true")
