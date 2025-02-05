if(WITH_TESTING)
  py_test(test_custom_raw_op_kernel_op_deprecated
          SRCS test_custom_raw_op_kernel_op_deprecated.py)
  set_tests_properties(test_custom_raw_op_kernel_op_deprecated
                       PROPERTIES TIMEOUT 180)
  if(NOT WIN32)
    # TODO(YuanRisheng) : Currently, we run this unittest by translating old ir to new ir, and it has bug that can't judge whether op_desc is a inplace op in windows.
    # We will fix it when abandoning translation in final state.
    if(WITH_GPU)
      py_test(test_inference_inplace SRCS test_inference_inplace.py)
      set_tests_properties(test_inference_inplace PROPERTIES TIMEOUT 180)
    endif()
  endif()
endif()
