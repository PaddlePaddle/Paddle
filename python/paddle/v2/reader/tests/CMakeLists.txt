add_test(NAME reader_tests
  COMMAND bash ${PROJ_ROOT}/python/paddle/v2/reader/tests/run_tests.sh
  ${PYTHON_EXECUTABLE})
