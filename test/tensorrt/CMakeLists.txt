file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(
    ${TEST_OP}
    MODULES
    ${TEST_OP}
    ENVS
    "PYTHONPATH=$PYTHONPATH:${PADDLE_BINARY_DIR}/python:/usr/local/TensorRT-8.5.3.1/python/"
  )
endforeach()
