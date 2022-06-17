file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach()
set_tests_properties(test_concat_mkldnn_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_conv3d_mkldnn_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_flags_mkldnn_ops_on_off PROPERTIES TIMEOUT 120)
