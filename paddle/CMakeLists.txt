# to limit the mobile dependencies
if (NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
  add_subdirectory(scripts)
  add_subdirectory(testing)
  set(PYTHON_TESTS_DIR ${PADDLE_BINARY_DIR}/python/paddle/fluid/tests CACHE INTERNAL "python tests directory")
endif()
add_subdirectory(fluid)
