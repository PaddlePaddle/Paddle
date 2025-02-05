if(WITH_IPU)
  file(
    GLOB TEST_OPS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "test_*.py")
  string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
  foreach(TEST_OP ${TEST_OPS})
    py_test_modules(${TEST_OP} MODULES ${TEST_OP})
    # set all UTs timeout to 200s
    set_tests_properties(${TEST_OP} PROPERTIES TIMEOUT 200)
  endforeach()

  set_tests_properties(test_conv_op_ipu PROPERTIES TIMEOUT 300)
  set_tests_properties(test_elementwise_x_op_ipu PROPERTIES TIMEOUT 600)
  set_tests_properties(test_reduce_x_op_ipu PROPERTIES TIMEOUT 800)
  set_tests_properties(test_save_load_ipu PROPERTIES TIMEOUT 600)
  set_tests_properties(test_activation_ops_ipu PROPERTIES TIMEOUT 600)
  set_tests_properties(test_unary_ops_ipu PROPERTIES TIMEOUT 600)
  set_tests_properties(test_greater_op_ipu PROPERTIES TIMEOUT 400)
  set_tests_properties(test_conv2d_transpose_op_ipu PROPERTIES TIMEOUT 300)
  set_tests_properties(test_model_parallel_ipu PROPERTIES TIMEOUT 300)

  add_subdirectory(custom_ops)

endif()
