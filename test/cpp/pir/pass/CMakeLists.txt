cc_test(
  pass_manager_test
  SRCS pass_manager_test.cc
  DEPS pir op_dialect_vjp phi common gtest)
