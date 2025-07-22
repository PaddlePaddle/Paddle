if(WITH_GPU
   AND (ARCH_BIN_CONTAINS_90 GREATER -1)
   AND (CUDA_VERSION GREATER_EQUAL 12.9))
  file(
    GLOB TEST_OPS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "test_*.py")
  string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

  foreach(TEST_OP ${TEST_OPS})
    py_test_modules(${TEST_OP} MODULES ${TEST_OP})
  endforeach()

endif()
