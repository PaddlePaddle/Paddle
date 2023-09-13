# file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
# string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

if(WITH_DISTRIBUTE)

  # NOTE(zyl): unittests WITH single card and WITHOUT timeout
  py_test_modules(test_matmul_rule MODULES test_matmul_rule)
  py_test_modules(test_embedding_rule MODULES test_embedding_rule)
  py_test_modules(test_replicated_rule MODULES test_replicated_rule)
  py_test_modules(test_softmax_rule MODULES test_softmax_rule)
  py_test_modules(test_split_rule MODULES test_split_rule)
  py_test_modules(test_transpose_rule MODULES test_transpose_rule)
  py_test_modules(test_elementwise_rule MODULES test_elementwise_rule)
  py_test_modules(test_cross_entropy_with_softmax_rule MODULES
                  test_cross_entropy_with_softmax_rule)
  py_test_modules(test_reduction_rule MODULES test_reduction_rule)
  py_test_modules(test_reshape_rule MODULES test_reshape_rule)
  py_test_modules(test_default_data_parallel_rule MODULES
                  test_default_data_parallel_rule)
  # End of unittests WITH single card WITHOUT timeout

endif()
