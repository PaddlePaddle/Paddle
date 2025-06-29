file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach()

set_tests_properties(test_setitem_appendix
                     PROPERTIES ENVIRONMENT "FLAGS_use_stride_kernel=1")
