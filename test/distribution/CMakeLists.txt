file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach()

set_pit_tests_properties()

if(WIN32)
  set_tests_properties(test_distribution_gamma PROPERTIES TIMEOUT 
                                                          5000)
  set_tests_properties(test_standard_gamma_op PROPERTIES TIMEOUT 
                                                          5000)
