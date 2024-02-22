if(WITH_GPU)
  nv_test(
    cuda_malloc_async_allocator_test
    SRCS cuda_malloc_async_allocator_test.cu
    DEPS cuda_graph_with_memory_pool)
  if(WITH_TESTING AND TEST cuda_malloc_async_allocator_test)
    set_tests_properties(
      cuda_malloc_async_allocator_test
      PROPERTIES ENVIRONMENT "FLAGS_use_cuda_malloc_async_allocator=true")
  endif()
endif()
