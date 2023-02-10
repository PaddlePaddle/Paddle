cc_test(
  type_support_test
  SRCS type_support_test.cc
  DEPS gtest)

cc_library(storage_uniquer SRCS storage_uniquer.cc)
