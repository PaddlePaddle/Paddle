paddle_test(
  if_op_test
  SRCS
  if_op_test.cc
  DEPS
  pir
  op_dialect_vjp
)

paddle_test(
  while_op_test
  SRCS
  while_op_test.cc
  DEPS
  pir
  op_dialect_vjp
)
