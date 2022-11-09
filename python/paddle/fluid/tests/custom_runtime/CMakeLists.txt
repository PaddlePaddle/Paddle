if(WITH_CUSTOM_DEVICE AND NOT WITH_GPU)
  set(PLUGIN_URL https://github.com/PaddlePaddle/PaddleCustomDevice.git)
  set(PLUGIN_TAG 0698428ddba21e6baecb690579f37c48896f7d56)

  file(
    GLOB TEST_OPS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "test_*.py")
  string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

  foreach(TEST_OP ${TEST_OPS})
    py_test(${TEST_OP}
            SRCS ${TEST_OP}.py ENVS FLAGS_allocator_strategy=naive_best_fit
                 PLUGIN_URL=${PLUGIN_URL} PLUGIN_TAG=${PLUGIN_TAG})
  endforeach()

  bash_test_modules(
    test_fleet_launch_custom_device
    START_BASH
    test_fleet_launch_custom_device.sh
    ENVS
    FLAGS_allocator_strategy=naive_best_fit
    PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR}
    PLUGIN_URL=${PLUGIN_URL}
    PLUGIN_TAG=${PLUGIN_TAG})

  set_tests_properties(test_custom_cpu_plugin PROPERTIES TIMEOUT 120)
  set_tests_properties(test_custom_cpu_profiler_plugin PROPERTIES TIMEOUT 120)
  set_tests_properties(test_fleet_launch_custom_device PROPERTIES TIMEOUT 120)
  set_tests_properties(test_custom_cpu_to_static PROPERTIES TIMEOUT 120)
endif()
