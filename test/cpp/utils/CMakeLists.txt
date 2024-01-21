if(NOT WITH_GFLAGS)
  paddle_test(flags_native_test SRCS flags_native_test.cc DEPS paddle_flags)
endif()

paddle_test(
  array_ref_test
  SRCS
  array_ref_test.cc
  DEPS
  gtest
  phi
  common)
paddle_test(
  small_vector_test
  SRCS
  small_vector_test.cc
  DEPS
  gtest
  phi
  common)
paddle_test(variant_test SRCS variant_test.cc DEPS gtest)
paddle_test(span_test SRCS span_test.cc DEPS gtest)
