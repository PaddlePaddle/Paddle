paddle_test(test_custom_kernel SRCS test_custom_kernel.cc)
paddle_test(test_dense_tensor SRCS test_dense_tensor.cc)
paddle_test(test_intrusive_ptr SRCS test_intrusive_ptr.cc)
paddle_test(test_type_info SRCS test_type_info.cc)
paddle_test(test_kernel_factory SRCS test_kernel_factory.cc)
paddle_test(test_sparse_coo_tensor SRCS test_sparse_coo_tensor.cc)
paddle_test(test_sparse_csr_tensor SRCS test_sparse_csr_tensor.cc)
paddle_test(test_op_utils SRCS test_op_utils.cc)
paddle_test(test_meta_fn_utils SRCS test_meta_fn_utils.cc)

paddle_test(test_ddim SRCS test_ddim.cc)
if(WITH_GPU)
  nv_test(
    test_dim
    SRCS test_dim.cu
    DEPS phi common)
elseif(WITH_ROCM)
  hip_test(
    test_dim
    SRCS test_dim.cu
    DEPS phi common)
endif()

paddle_test(selected_rows_test SRCS test_selected_rows.cc)
if(WITH_TESTING AND TEST selected_rows_test)
  set_tests_properties(selected_rows_test PROPERTIES TIMEOUT 120)
endif()
if(NOT WIN32)
  paddle_test(test_rw_lock SRCS test_rw_lock.cc)
endif()
paddle_test(test_string_tensor SRCS test_string_tensor.cc)
paddle_test(unroll_array_ops_test SRCS unroll_array_ops_test.cc)

paddle_test(test_tensor_array SRCS test_tensor_array.cc)

if(WITH_GPU)
  nv_test(
    test_mixed_vector
    SRCS test_mixed_vector.cc test_mixed_vector.cu
    DEPS place phi common tensor)
elseif(WITH_ROCM)
  hip_test(
    test_mixed_vector
    SRCS test_mixed_vector.cc test_mixed_vector.cu
    DEPS place phi common tensor)
else()
  paddle_test(test_mixed_vector SRCS test_mixed_vector.cc)
endif()
