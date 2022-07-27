if(WITH_CUSTOM_DEVICE)
  file(
    GLOB TEST_OPS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "test_*.py")
  string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

  list(REMOVE_ITEM TEST_OPS "test_collective_process_group_xccl")
  foreach(TEST_OP ${TEST_OPS})
    py_test(${TEST_OP} SRCS ${TEST_OP}.py)
    set_tests_properties(${TEST_OP} PROPERTIES TIMEOUT 120)
  endforeach()

  bash_test_modules(
    test_fleet_launch_custom_device START_BASH
    test_fleet_launch_custom_device.sh ENVS
    PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR})
endif()
