set(COMMON_API_TEST_DEPS phi common)

if(WITH_GPU)
  nv_test(
    test_phi_tensor
    SRCS test_phi_tensor.cc
    DEPS glog ${COMMON_API_TEST_DEPS})
  nv_test(
    test_allocator
    SRCS test_allocator.cu
    DEPS device_context phi common)
  nv_test(
    test_cuda_stream
    SRCS test_cuda_stream.cu
    DEPS phi common)
  nv_test(
    test_from_blob
    SRCS test_from_blob.cc
    DEPS ${COMMON_API_TEST_DEPS})
elseif(WITH_ROCM)
  hip_test(
    test_phi_tensor
    SRCS test_phi_tensor.cc
    DEPS glog ${COMMON_API_TEST_DEPS})
  hip_test(
    test_allocator
    SRCS test_allocator.cu
    DEPS device_context phi common)
  hip_test(
    test_cuda_stream
    SRCS test_cuda_stream.cu
    DEPS phi common)
  hip_test(
    test_from_blob
    SRCS test_from_blob.cc
    DEPS ${COMMON_API_TEST_DEPS})
else()
  cc_test(
    test_phi_tensor
    SRCS test_phi_tensor.cc
    DEPS glog ${COMMON_API_TEST_DEPS})
  cc_test(
    test_from_blob
    SRCS test_from_blob.cc
    DEPS ${COMMON_API_TEST_DEPS})
endif()

cc_test(
  test_phi_exception
  SRCS test_phi_exception.cc
  DEPS gtest)

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
