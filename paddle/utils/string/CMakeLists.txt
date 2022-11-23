cc_library(
<<<<<<< HEAD
  stringpiece
  SRCS piece.cc
  DEPS flags)
cc_library(
=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
  pretty_log
  SRCS pretty_log.cc
  DEPS flags)
cc_library(
  string_helper
  SRCS string_helper.cc
  DEPS flags)
cc_test(
<<<<<<< HEAD
  stringpiece_test
  SRCS piece_test.cc
  DEPS stringpiece gflags)
cc_test(
=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
  stringprintf_test
  SRCS printf_test.cc
  DEPS gflags)
cc_test(to_string_test SRCS to_string_test.cc)
cc_test(split_test SRCS split_test.cc)
cc_test(
  string_helper_test
  SRCS string_helper_test.cc
  DEPS string_helper)
