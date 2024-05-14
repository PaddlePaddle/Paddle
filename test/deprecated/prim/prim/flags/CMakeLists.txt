file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP} ENVS ${GC_ENVS})
endforeach()

if(WITH_CINN)
  set_tests_properties(test_prim_flags_case_deprecated
                       PROPERTIES LABELS "RUN_TYPE=CINN")
  set_tests_properties(test_prim_flags_case_deprecated PROPERTIES TIMEOUT 300)
endif()
