if(WITH_GPU)
  nv_test(
    test_phi_tensor
    SRCS test_phi_tensor.cc
    DEPS phi_tensor glog)
elseif(WITH_ROCM)
  hip_test(
    test_phi_tensor
    SRCS test_phi_tensor.cc
    DEPS phi_tensor glog)
else()
  cc_test(
    test_phi_tensor
    SRCS test_phi_tensor.cc
    DEPS phi_tensor glog)
endif()

cc_test(
  test_phi_exception
  SRCS test_phi_exception.cc
  DEPS gtest)

set(COMMON_API_TEST_DEPS phi_tensor phi_api phi_api_utils)
cc_test(
  test_to_api
  SRCS test_to_api.cc
  DEPS ${COMMON_API_TEST_DEPS})
cc_test(
  test_slice_api
  SRCS test_slice_api.cc
  DEPS ${COMMON_API_TEST_DEPS})
cc_test(
  test_scale_benchmark
  SRCS test_scale_benchmark.cc
  DEPS ${COMMON_API_TEST_DEPS})
cc_test(
  test_data_transform
  SRCS test_data_transform.cc
  DEPS ${COMMON_API_TEST_DEPS})
cc_test(
  test_strings_empty_api
  SRCS test_strings_empty_api.cc
  DEPS ${COMMON_API_TEST_DEPS})
cc_test(
  test_strings_lower_upper_api
  SRCS test_strings_lower_upper_api.cc
  DEPS ${COMMON_API_TEST_DEPS})
