file(GLOB TEST_INTERP_CASES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_INTERP_CASES "${TEST_INTERP_CASES}")

foreach(target ${TEST_INTERP_CASES})
  py_test_modules(${target} MODULES ${target} ENVS FLAGS_use_stream_safe_cuda_allocator=false)
endforeach()

foreach(target ${TEST_INTERP_CASES})
  py_test_modules(${target}_fast_gc MODULES ${target} ENVS FLAGS_use_stream_safe_cuda_allocator=true)
endforeach()
