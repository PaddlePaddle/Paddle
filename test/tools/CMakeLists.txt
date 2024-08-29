file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

if((NOT WITH_COVERAGE) AND (LINUX))
  foreach(TEST_OP ${TEST_OPS})
    py_test_modules(${TEST_OP} MODULES ${TEST_OP} ENVS
                    PYTHONPATH=${PADDLE_SOURCE_DIR}/tools)
  endforeach()

  set_tests_properties(test_sampcd_processor PROPERTIES TIMEOUT 300)
  set_tests_properties(test_type_checking PROPERTIES TIMEOUT 200)
endif()
