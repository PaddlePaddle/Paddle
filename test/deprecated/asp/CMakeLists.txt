file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach()

set_tests_properties(test_asp_pruning_dynamic PROPERTIES TIMEOUT 30)
set_tests_properties(test_asp_pruning_static_deprecated PROPERTIES TIMEOUT 30)
set_tests_properties(test_asp_optimize_dynamic PROPERTIES TIMEOUT 30)
set_tests_properties(test_asp_optimize_static_deprecated PROPERTIES TIMEOUT 30)
