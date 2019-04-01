if (NOT WITH_ARM_CPU)
  add_subdirectory(scripts)
  add_subdirectory(testing)
  set(PYTHON_TESTS_DIR ${PADDLE_BINARY_DIR}/python/paddle/fluid/tests CACHE INTERNAL "python tests directory")
endif(NOT WITH_ARM_CPU)

add_subdirectory(fluid)
