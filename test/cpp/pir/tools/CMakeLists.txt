cc_library(
  test_dialect
  SRCS test_dialect.cc test_op.cc test_trait.cc test_interface.cc
  DEPS pir gtest)
