cc_library(
<<<<<<< HEAD
=======
  stringpiece
  SRCS piece.cc
  DEPS flags)
cc_library(
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
  pretty_log
  SRCS pretty_log.cc
  DEPS flags)
cc_library(
  string_helper
  SRCS string_helper.cc
  DEPS flags)
cc_test(
<<<<<<< HEAD
=======
  stringpiece_test
  SRCS piece_test.cc
  DEPS stringpiece gflags)
cc_test(
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
  stringprintf_test
  SRCS printf_test.cc
  DEPS gflags)
cc_test(to_string_test SRCS to_string_test.cc)
cc_test(split_test SRCS split_test.cc)
cc_test(
  string_helper_test
  SRCS string_helper_test.cc
  DEPS string_helper)
