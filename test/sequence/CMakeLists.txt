file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach()

foreach(PIR_COVERAGE_TEST ${PIR_COVERAGE_TESTS})
  py_test_modules(${PIR_COVERAGE_TEST}_pir MODULES ${PIR_COVERAGE_TEST} ENVS
                  FLAGS_enable_pir_in_executor=true)
  set_tests_properties(${PIR_COVERAGE_TEST}_pir PROPERTIES TIMEOUT 120)
  message(STATUS "PIR Copied OpTest: ${PIR_COVERAGE_TEST}_pir in sequence test")
endforeach()
