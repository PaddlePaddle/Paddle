file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

if(NOT WITH_HETERPS OR WITH_PSLIB)
  list(REMOVE_ITEM TEST_OPS "test_gpubox_ps")
endif()

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP})
  list(APPEND TEST_OPS ${TEST_OP})
  set_tests_properties(${TEST_OP} PROPERTIES TIMEOUT 120)
endforeach()

if(WITH_HETERPS AND NOT WITH_PSLIB)
  set_tests_properties(test_gpubox_ps PROPERTIES LABELS "RUN_TYPE=GPUPS")
endif()
