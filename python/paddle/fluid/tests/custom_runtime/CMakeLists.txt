if(WITH_CUSTOM_DEVICE AND NOT WITH_GPU)
  set(PLUGIN_URL https://github.com/PaddlePaddle/PaddleCustomDevice.git)
  set(PLUGIN_TAG d5e5ac1d8e9f7588d4c2998bb3b5ffc66f65af2e)

  file(
    GLOB TEST_OPS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "test_*.py")
  string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

  list(REMOVE_ITEM TEST_OPS test_collective_process_group_xccl)
  foreach(TEST_OP ${TEST_OPS})
    py_test(${TEST_OP} SRCS ${TEST_OP}.py ENVS PLUGIN_URL=${PLUGIN_URL}
                            PLUGIN_TAG=${PLUGIN_TAG})
  endforeach()

  bash_test_modules(
    test_fleet_launch_custom_device
    START_BASH
    test_fleet_launch_custom_device.sh
    ENVS
    PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR}
    PLUGIN_URL=${PLUGIN_URL}
    PLUGIN_TAG=${PLUGIN_TAG})

  set_tests_properties(test_custom_cpu_plugin PROPERTIES TIMEOUT 120)
  set_tests_properties(test_custom_cpu_profiler_plugin PROPERTIES TIMEOUT 120)
  set_tests_properties(test_fleet_launch_custom_device PROPERTIES TIMEOUT 120)
endif()
