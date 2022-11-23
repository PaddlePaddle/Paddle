<<<<<<< HEAD
if(WITH_CUSTOM_DEVICE)
=======
if(WITH_CUSTOM_DEVICE AND NOT WITH_GPU)
  set(PLUGIN_URL https://github.com/PaddlePaddle/PaddleCustomDevice.git)
  set(PLUGIN_TAG develop)

>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
  file(
    GLOB TEST_OPS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "test_*.py")
  string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

<<<<<<< HEAD
  list(REMOVE_ITEM TEST_OPS "test_collective_process_group_xccl")
  foreach(TEST_OP ${TEST_OPS})
    py_test(${TEST_OP} SRCS ${TEST_OP}.py)
  endforeach()
=======
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
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
endif()
