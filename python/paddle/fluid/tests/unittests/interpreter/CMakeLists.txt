file(GLOB TEST_INTERP_CASES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_INTERP_CASES "${TEST_INTERP_CASES}")

foreach(target ${TEST_INTERP_CASES})
  py_test_modules(${target} MODULES ${target} ENVS FLAGS_host_trace_level=10 FLAGS_static_executor_perfstat_filepath=./perfstat FLAGS_allocator_strategy=auto_growth FLAGS_use_stream_safe_cuda_allocator=true FLAGS_fast_eager_deletion_mode=false FLAGS_eager_delete_tensor_gb=0)
  py_test_modules(${target}_non_eager_deletion MODULES ${target} ENVS FLAGS_allocator_strategy=auto_growth FLAGS_use_stream_safe_cuda_allocator=true FLAGS_fast_eager_deletion_mode=false FLAGS_eager_delete_tensor_gb=0.000001) 
  py_test_modules(${target}_fast_gc MODULES ${target} ENVS FLAGS_allocator_strategy=auto_growth FLAGS_use_stream_safe_cuda_allocator=true FLAGS_fast_eager_deletion_mode=true FLAGS_eager_delete_tensor_gb=0)
  py_test_modules(${target}_fast_gc_non_eager_deletion MODULES ${target} ENVS FLAGS_allocator_strategy=auto_growth FLAGS_use_stream_safe_cuda_allocator=true FLAGS_fast_eager_deletion_mode=true FLAGS_eager_delete_tensor_gb=0.000001)
endforeach()
