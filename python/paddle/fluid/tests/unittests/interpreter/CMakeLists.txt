file(GLOB TEST_INTERP_CASES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_INTERP_CASES "${TEST_INTERP_CASES}")

foreach(target ${TEST_INTERP_CASES})
  py_test_modules(${target} MODULES ${target})
endforeach()

set_tests_properties(test_standalone_executor PROPERTIES ENVIRONMENT FLAGS_use_stream_safe_cuda_allocator=true)
