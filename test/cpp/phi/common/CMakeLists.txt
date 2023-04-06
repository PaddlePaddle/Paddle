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
  DEPS phi)
cc_test(
  phi_test_int_array
  SRCS test_int_array.cc
  DEPS phi)
cc_test(
  phi_test_scalar_cpu
  SRCS test_scalar.cc
  DEPS phi)
if(WITH_GPU)
  nv_test(
    phi_test_scalar
    SRCS test_scalar.cu
    DEPS phi)
  nv_test(
    transform_test
    SRCS transform_test.cu
    DEPS memory place phi)
endif()
if(WITH_ROCM)
  hip_test(
    phi_test_scalar
    SRCS test_scalar.cu
    DEPS phi)
  hip_test(
    transform_test
    SRCS transform_test.cu
    DEPS memory place phi)
endif()
