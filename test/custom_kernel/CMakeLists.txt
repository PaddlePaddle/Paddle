file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

set(CUSTOM_ENVS
    PADDLE_SOURCE_DIR=${PADDLE_SOURCE_DIR}
    PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR}
    CUSTOM_DEVICE_ROOT=${CMAKE_BINARY_DIR}/python/paddle/fluid/tests/custom_kernel
)

foreach(TEST_OP ${TEST_OPS})
  py_test(${TEST_OP} SRCS ${TEST_OP}.py ENVS ${CUSTOM_ENVS})
endforeach()
