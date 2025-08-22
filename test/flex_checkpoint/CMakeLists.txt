file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(TEST_OP ${TEST_OPS})
  if(${TEST_OP} STREQUAL "test_strategy_conversion")
    set(WORKFLOW_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_OP}.py)

    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} ${WORKFLOW_SCRIPT} --list_tests
      OUTPUT_VARIABLE TEST_CASE_LIST
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REPLACE "\n" ";" TEST_CASE_LIST "${TEST_CASE_LIST}")

    foreach(TEST_CASE ${TEST_CASE_LIST})
      string(REPLACE "__main__.TestStrategyConversion.test_" "" TEST_CASE_ALIAS
                     ${TEST_CASE})

      add_test(NAME ${TEST_OP}.${TEST_CASE_ALIAS}
               COMMAND ${PYTHON_EXECUTABLE} -m unittest ${TEST_CASE})
    endforeach()
  else()
    py_test_modules(${TEST_OP} MODULES ${TEST_OP})
  endif()
endforeach()

set(GPU_ONLY_DISTRIBUTED_TESTS test_sharded_state_dict test_strategy_conversion)

if(NOT (WITH_DISTRIBUTE AND WITH_GPU))
  get_property(
    ALL_TESTS
    DIRECTORY
    PROPERTY TESTS)
  foreach(CURRENT_TEST_NAME ${ALL_TESTS})
    foreach(SUITE_NAME ${GPU_ONLY_DISTRIBUTED_TESTS})
      if("${CURRENT_TEST_NAME}" STREQUAL "${SUITE_NAME}"
         OR "${CURRENT_TEST_NAME}" MATCHES "^${SUITE_NAME}\\.")
        message(STATUS "Disabling GPU/Dist test: ${CURRENT_TEST_NAME}")
        set_tests_properties("${CURRENT_TEST_NAME}" PROPERTIES DISABLED TRUE)
      endif()
    endforeach()
  endforeach()
endif()
