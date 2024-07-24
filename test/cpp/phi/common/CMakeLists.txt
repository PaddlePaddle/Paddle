cc_test(
  phi_test_backend
  SRCS test_backend.cc
  DEPS gtest)
cc_test(
  phi_test_data_layout
  SRCS test_data_layout.cc
  DEPS gtest)
cc_test(
  phi_test_data_type
  SRCS test_data_type.cc
  DEPS gtest)
cc_test(
  phi_test_place
  SRCS test_place.cc
  DEPS phi common)
cc_test(
  phi_test_int_array
  SRCS test_int_array.cc
  DEPS phi common)
cc_test(
  phi_test_scalar_cpu
  SRCS test_scalar.cc
  DEPS phi common)
if(WITH_GPU)
  nv_test(
    phi_test_scalar
    SRCS test_scalar.cu
    DEPS phi common)
  nv_test(
    transform_test
    SRCS transform_test.cu
    DEPS phi common)
endif()
if(WITH_ROCM)
  hip_test(
    phi_test_scalar
    SRCS test_scalar.cu
    DEPS phi common)
  hip_test(
    transform_test
    SRCS transform_test.cu
    DEPS phi common)
endif()
