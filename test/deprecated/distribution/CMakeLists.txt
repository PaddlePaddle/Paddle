file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

# test_distribution has been tested in test/distribution dir
list(REMOVE_ITEM TEST_OPS test_distribution)

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach()

set_tests_properties(test_distribution_lkj_cholesky_static PROPERTIES TIMEOUT
                                                                      120)

set_pir_tests_properties()
