file(
  GLOB TEST_INTERP_CASES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
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
  test_standalone_executor_stats MODULES test_standalone_executor ENVS
  FLAGS_host_trace_level=10 FLAGS_static_executor_perfstat_filepath=./perfstat)

set_tests_properties(test_standalone_cross_step_overlap PROPERTIES TIMEOUT 30)
