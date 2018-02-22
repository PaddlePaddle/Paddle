if (NOT APPLE)
  # The Mac OS X backend will not be able to function correctly if Python is
  # not installed as a framework.
  py_test(test_ploter SRCS test_ploter.py)
endif()
