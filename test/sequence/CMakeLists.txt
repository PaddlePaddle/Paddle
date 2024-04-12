file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

# The following unittest is now in deprecated dir, we can delete this code when we move it from deprecated dir to this dir
list(REMOVE_ITEM TEST_OPS test_sequence_conv)
list(REMOVE_ITEM TEST_OPS test_sequence_pool)

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach()

set(PIR_COVERAGE_TESTS test_sequence_mask)

foreach(PIR_COVERAGE_TEST ${PIR_COVERAGE_TESTS})
  py_test_modules(${PIR_COVERAGE_TEST}_pir MODULES ${PIR_COVERAGE_TEST} ENVS
                  FLAGS_enable_pir_in_executor=true)
  set_tests_properties(${PIR_COVERAGE_TEST}_pir PROPERTIES TIMEOUT 120)
  message(STATUS "PIR Copied OpTest: ${PIR_COVERAGE_TEST}_pir in sequence test")
endforeach()
