cc_library(
  pretty_log
  SRCS pretty_log.cc
  DEPS phi common)
cc_library(
  string_helper
  SRCS string_helper.cc
  DEPS phi common)
cc_test(
  stringprintf_test
  SRCS printf_test.cc
  DEPS phi common)
cc_test(to_string_test SRCS to_string_test.cc)
cc_test(split_test SRCS split_test.cc)
cc_test(
  string_helper_test
  SRCS string_helper_test.cc
  DEPS string_helper)
