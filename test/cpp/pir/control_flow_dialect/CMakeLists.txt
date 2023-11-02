cc_test_old(
  if_op_test
  SRCS
  if_op_test.cc
  DEPS
  pir
  dialect
  gtest)

cc_test_old(
  while_op_test
  SRCS
  while_op_test.cc
  DEPS
  pir
  dialect
  gtest)
