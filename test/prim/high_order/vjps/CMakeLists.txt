file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP} ENVS FLAGS_enable_pir_api=true
                  FLAGS_prim_enable_dynamic=true)
endforeach()

set_tests_properties(test_binary_func PROPERTIES TIMEOUT 90)
set_tests_properties(test_unary_func PROPERTIES TIMEOUT 90)
