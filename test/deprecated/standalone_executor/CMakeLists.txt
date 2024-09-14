file(
  GLOB TEST_INTERP_CASES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
list(REMOVE_ITEM TEST_INTERP_CASES "test_standalone_custom_event.py")
string(REPLACE ".py" "" TEST_INTERP_CASES "${TEST_INTERP_CASES}")

foreach(target ${TEST_INTERP_CASES})
  py_test_modules(${target} MODULES ${target})
endforeach()

# These UTs are to temporarily test static build for standalone_executor, will be removed after static build is enabled by default.
set(STATIC_BUILD_TESTS test_standalone_cuda_graph_multi_stream_deprecated)

foreach(STATIC_BUILD_TEST ${STATIC_BUILD_TESTS})
  py_test_modules(
    ${STATIC_BUILD_TEST}_static_build_deprecated MODULES ${STATIC_BUILD_TEST}
    ENVS FLAGS_new_executor_static_build=true)
endforeach()

set_tests_properties(test_standalone_executor_aot_choose_kernel_deprecated
                     PROPERTIES TIMEOUT 60)
