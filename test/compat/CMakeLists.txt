file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(TEST_OP ${TEST_OPS})
  if(TEST_OP STREQUAL "test_scalar_type_caster")
    py_test_modules(
      ${TEST_OP} MODULES ${TEST_OP} ENVS PADDLE_SOURCE_DIR=${PADDLE_SOURCE_DIR}
      PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR})
  else()
    py_test_modules(${TEST_OP} MODULES ${TEST_OP})
  endif()
endforeach()
