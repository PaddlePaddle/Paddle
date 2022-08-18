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
  DEPS phi_place)
cc_test(
  phi_test_int_array
  SRCS test_int_array.cc
  DEPS int_array api_int_array phi phi_api)
if(WITH_GPU)
  nv_test(
    phi_test_scalar
    SRCS test_scalar.cu
    DEPS scalar api_scalar)
endif()
if(WITH_ROCM)
  hip_test(
    phi_test_scalar
    SRCS test_scalar.cu
    DEPS scalar api_scalar)
endif()
