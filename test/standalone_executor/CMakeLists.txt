file(
  GLOB TEST_INTERP_CASES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
list(REMOVE_ITEM TEST_INTERP_CASES "test_standalone_custom_event.py")
string(REPLACE ".py" "" TEST_INTERP_CASES "${TEST_INTERP_CASES}")

foreach(target ${TEST_INTERP_CASES})
  py_test_modules(${target} MODULES ${target})
endforeach()

py_test_modules(
  test_standalone_executor_no_fast_gc MODULES test_standalone_executor ENVS
  FLAGS_fast_eager_deletion_mode=false)

py_test_modules(
  test_standalone_executor_sequential_run MODULES test_standalone_executor ENVS
  FLAGS_new_executor_sequential_run=true)

py_test_modules(
  test_standalone_executor_serial_run MODULES test_standalone_executor ENVS
  FLAGS_new_executor_serial_run=true)

py_test_modules(
  test_standalone_executor_log_deps MODULES test_standalone_executor ENVS
  GLOG_v=1 FLAGS_executor_log_deps_every_microseconds=1000)

py_test_modules(
  test_standalone_executor_stats MODULES test_standalone_executor ENVS
  FLAGS_host_trace_level=10 FLAGS_static_executor_perfstat_filepath=./perfstat)

# These UTs are to temporarily test static build for standalone_executor, will be removed after static build is enabled by default.
set(STATIC_BUILD_TESTS
    test_standalone_controlflow test_standalone_custom_stream
    test_standalone_custom_event test_standalone_multiply_write
    test_standalone_executor)

foreach(STATIC_BUILD_TEST ${STATIC_BUILD_TESTS})
  py_test_modules(
    ${STATIC_BUILD_TEST}_static_build MODULES ${STATIC_BUILD_TEST} ENVS
    FLAGS_new_executor_static_build=true)
endforeach()

set_tests_properties(test_standalone_cross_step_overlap PROPERTIES TIMEOUT 30)
