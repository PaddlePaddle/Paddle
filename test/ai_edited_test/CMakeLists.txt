file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

# Copy test .py files from source to build directory so test_runner.py can find them
file(GLOB TEST_PY_FILES "${CMAKE_CURRENT_SOURCE_DIR}/test_*.py")
file(COPY ${TEST_PY_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach()
