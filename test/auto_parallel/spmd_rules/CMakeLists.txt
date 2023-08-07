# file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
# string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

if(WITH_DISTRIBUTE AND WITH_GPU)

  # NOTE(zyl): unittests WITH single card and WITHOUT timeout
  py_test_modules(test_matmul_rule MODULES test_matmul_rule)
  py_test_modules(test_matmul_rule MODULES test_embedding_rule)
  py_test_modules(test_matmul_rule MODULES test_replicated_rule)
  py_test_modules(test_matmul_rule MODULES test_softmax_rule)
  py_test_modules(test_split_rule MODULES test_split_rule)
  py_test_modules(test_transpose_rule MODULES test_transpose_rule)
  # End of unittests WITH single card WITHOUT timeout

endif()
