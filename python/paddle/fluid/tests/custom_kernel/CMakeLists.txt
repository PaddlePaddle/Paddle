if(NOT WIN32 AND NOT APPLE)
    py_test(test_custom_kernel_dot SRCS test_custom_kernel_dot.py)
endif()
