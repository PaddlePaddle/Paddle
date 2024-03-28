cinn_cc_library(
  cinn_runtime_dialect
  SRCS
  runtime_dialect.cc
  jit_kernel_op.cc
  DEPS
  cinn_op_dialect
  pir)
