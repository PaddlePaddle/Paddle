if(NOT WITH_GFLAGS)
  cc_library(paddle_flags SRCS flags_native.cc)
  if(CINN_ONLY)
    return()
  endif()
  cc_test(
    flags_native_test
    SRCS flags_native_test.cc
    DEPS paddle_flags)
endif()

add_subdirectory(string)

cc_test(
  array_ref_test
  SRCS array_ref_test.cc
  DEPS gtest phi)
cc_test(
  small_vector_test
  SRCS small_vector_test.cc
  DEPS gtest phi)
cc_test(
  variant_test
  SRCS variant_test.cc
  DEPS gtest)
cc_test(
  span_test
  SRCS span_test.cc
  DEPS gtest)

if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
  cc_library(
    pybind_util
    SRCS pybind.cc
    DEPS phi)
endif()
