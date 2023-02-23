add_subdirectory(string)

cc_test(
  array_ref_test
  SRCS array_ref_test.cc
  DEPS gtest gflags)
cc_test(
  small_vector_test
  SRCS small_vector_test.cc
  DEPS gtest gflags)
cc_test(
  variant_test
  SRCS variant_test.cc
  DEPS gtest)

if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
  cc_library(
    pybind_util
    SRCS pybind.cc
    DEPS phi_tensor_raw)
endif()
