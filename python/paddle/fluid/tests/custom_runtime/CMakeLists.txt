if(WITH_CUSTOM_DEVICE)
  file(
    GLOB TEST_OPS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "test_*.py")
  string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

  foreach(TEST_OP ${TEST_OPS})
    py_test(${TEST_OP} SRCS ${TEST_OP}.py)
  endforeach()

  set_tests_properties(test_custom_cpu_plugin PROPERTIES TIMEOUT 120)
  set_tests_properties(test_collective_process_group_xccl PROPERTIES TIMEOUT
                                                                     120)
endif()
