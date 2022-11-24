if(WITH_GPU)
  nv_test(
    test_phi_tensor
<<<<<<< HEAD
    SRCS test_pten_tensor.cc
=======
    SRCS test_phi_tensor.cc
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    DEPS phi_tensor glog)
elseif(WITH_ROCM)
  hip_test(
    test_phi_tensor
<<<<<<< HEAD
    SRCS test_pten_tensor.cc
=======
    SRCS test_phi_tensor.cc
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    DEPS phi_tensor glog)
else()
  cc_test(
    test_phi_tensor
<<<<<<< HEAD
    SRCS test_pten_tensor.cc
=======
    SRCS test_phi_tensor.cc
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    DEPS phi_tensor glog)
endif()

cc_test(
  test_phi_exception
<<<<<<< HEAD
  SRCS test_pten_exception.cc
=======
  SRCS test_phi_exception.cc
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  DEPS gtest)

set(COMMON_API_TEST_DEPS phi_tensor phi_api phi_api_utils)
cc_test(
<<<<<<< HEAD
  test_mean_api
  SRCS test_mean_api.cc
  DEPS ${COMMON_API_TEST_DEPS})
cc_test(
  test_dot_api
  SRCS test_dot_api.cc
  DEPS ${COMMON_API_TEST_DEPS})
cc_test(
  test_matmul_api
  SRCS test_matmul_api.cc
  DEPS ${COMMON_API_TEST_DEPS})
cc_test(
  test_empty_api
  SRCS test_empty_api.cc
  DEPS ${COMMON_API_TEST_DEPS})
cc_test(
  test_fill_api
  SRCS test_fill_api.cc
  DEPS ${COMMON_API_TEST_DEPS} api_scalar)
cc_test(
  test_elementwise_api
  SRCS test_elementwise_api.cc
  DEPS ${COMMON_API_TEST_DEPS})
cc_test(
  test_embedding_api
  SRCS test_embedding_api.cc
  DEPS ${COMMON_API_TEST_DEPS})
cc_test(
  test_cast_api
  SRCS test_cast_api.cc
  DEPS ${COMMON_API_TEST_DEPS})
cc_test(
  test_reshape_api
  SRCS test_reshape_api.cc
  DEPS ${COMMON_API_TEST_DEPS})
cc_test(
=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  test_to_api
  SRCS test_to_api.cc
  DEPS ${COMMON_API_TEST_DEPS})
cc_test(
  test_slice_api
  SRCS test_slice_api.cc
  DEPS ${COMMON_API_TEST_DEPS})
cc_test(
<<<<<<< HEAD
  test_sum_api
  SRCS test_sum_api.cc
  DEPS ${COMMON_API_TEST_DEPS})
cc_test(
  test_scale_api
  SRCS test_scale_api.cc
  DEPS ${COMMON_API_TEST_DEPS} api_scalar)
cc_test(
=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  test_scale_benchmark
  SRCS test_scale_benchmark.cc
  DEPS ${COMMON_API_TEST_DEPS})
cc_test(
<<<<<<< HEAD
  test_conj_api
  SRCS test_conj_api.cc
  DEPS ${COMMON_API_TEST_DEPS})
cc_test(
  test_concat_api
  SRCS test_concat_api.cc
  DEPS ${COMMON_API_TEST_DEPS})
cc_test(
  test_split_api
  SRCS test_split_api.cc
  DEPS ${COMMON_API_TEST_DEPS})
cc_test(
=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  test_data_transform
  SRCS test_data_transform.cc
  DEPS ${COMMON_API_TEST_DEPS})
cc_test(
<<<<<<< HEAD
  test_sparse_utils_api
  SRCS test_sparse_utils_api.cc
  DEPS ${COMMON_API_TEST_DEPS})
cc_test(
  test_sparse_conv_api
  SRCS test_sparse_conv_api.cc
  DEPS ${COMMON_API_TEST_DEPS})
cc_test(
=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  test_strings_empty_api
  SRCS test_strings_empty_api.cc
  DEPS ${COMMON_API_TEST_DEPS})
cc_test(
  test_strings_lower_upper_api
  SRCS test_strings_lower_upper_api.cc
  DEPS ${COMMON_API_TEST_DEPS})
