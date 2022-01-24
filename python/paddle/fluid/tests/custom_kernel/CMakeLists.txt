# WIN32 is not supported
if(NOT WIN32)
    py_test(test_custom_kernel_dot SRCS test_custom_kernel_dot.py)
endif()
