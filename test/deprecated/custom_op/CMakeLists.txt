if(WITH_TESTING)
  py_test(test_custom_raw_op_kernel_op SRCS test_custom_raw_op_kernel_op.py)
  set_tests_properties(test_custom_raw_op_kernel_op PROPERTIES TIMEOUT 180)

  py_test(test_custom_cast_op_jit SRCS test_custom_cast_op_jit.py)
  set_tests_properties(test_custom_cast_op_jit PROPERTIES TIMEOUT 180)
endif()
