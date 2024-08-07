if(WITH_TESTING)
  py_test(test_custom_raw_op_kernel_op_deprecated
          SRCS test_custom_raw_op_kernel_op_deprecated.py)
  set_tests_properties(test_custom_raw_op_kernel_op_deprecated
                       PROPERTIES TIMEOUT 180)
endif()
