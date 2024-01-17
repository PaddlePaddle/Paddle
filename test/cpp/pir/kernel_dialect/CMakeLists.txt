paddle_test(
  ir_kernel_dialect_pass_test
  SRCS
  ir_kernel_dialect_pass_test.cc
  DEPS
  program_translator
  common
  gtest)
