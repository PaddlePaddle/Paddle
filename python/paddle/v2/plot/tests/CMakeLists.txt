if (NOT APPLE)
  # The Mac OS X backend will not be able to function correctly if Python is
  # not installed as a framework.
  add_python_test(test_ploter test_ploter.py)
endif()
