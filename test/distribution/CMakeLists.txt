file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

if(WIN32)
  # standard_gamma_grap_op crash in win
  list(REMOVE_ITEM TEST_OPS test_distribution_gamma)
  list(REMOVE_ITEM TEST_OPS test_standard_gamma_op)
endif()

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach()

set_pit_tests_properties()
