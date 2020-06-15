# TODO: Fix this unittest failed on Windows
if(NOT WIN32)
    py_test(decorator_test SRCS decorator_test.py)
endif()
