cc_test(
  test_custom_kernel
  SRCS test_custom_kernel.cc
  DEPS custom_kernel scalar)
cc_test(
  test_dense_tensor
  SRCS test_dense_tensor.cc
  DEPS dense_tensor)
cc_test(test_intrusive_ptr SRCS test_intrusive_ptr.cc)
cc_test(test_type_info SRCS test_type_info.cc)
cc_test(
  test_kernel_factory
  SRCS test_kernel_factory.cc
  DEPS kernel_factory phi)
cc_test(
  test_sparse_coo_tensor
  SRCS test_sparse_coo_tensor.cc
  DEPS dense_tensor sparse_coo_tensor)
cc_test(
  test_sparse_csr_tensor
  SRCS test_sparse_csr_tensor.cc
  DEPS dense_tensor sparse_csr_tensor)
cc_test(
  test_op_utils
  SRCS test_op_utils.cc
  DEPS op_compat_infos)
cc_test(
  test_meta_fn_utils
  SRCS test_meta_fn_utils.cc
  DEPS dense_tensor wrapped_infermeta infermeta infermeta_utils)

cc_test(
  test_ddim
  SRCS test_ddim.cc
  DEPS ddim)
if(WITH_GPU)
  nv_test(
    test_dim
    SRCS test_dim.cu
    DEPS ddim)
elseif(WITH_ROCM)
  hip_test(
    test_dim
    SRCS test_dim.cu
    DEPS ddim)
endif()

cc_test(
  selected_rows_test
  SRCS test_selected_rows.cc
  DEPS selected_rows)
if(WITH_TESTING AND TEST selected_rows_test)
  set_tests_properties(selected_rows_test PROPERTIES TIMEOUT 120)
endif()
if(NOT WIN32)
  cc_test(test_rw_lock SRCS test_rw_lock.cc)
endif()
cc_test(
  test_string_tensor
  SRCS test_string_tensor.cc
  DEPS string_tensor)
cc_test(unroll_array_ops_test SRCS unroll_array_ops_test.cc)

cc_test(
  test_tensor_array
  SRCS test_tensor_array.cc
  DEPS tensor_array)
