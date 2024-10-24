set(ALLOCATOR_SRCS
    allocator.cc
    cpu_allocator.cc
    aligned_allocator.cc
    buffered_allocator.cc
    best_fit_allocator.cc
    naive_best_fit_allocator.cc
    allocator_strategy.cc
    allocator_facade.cc
    auto_growth_best_fit_allocator.cc
    auto_growth_best_fit_allocator_v2.cc
    virtual_memory_auto_growth_best_fit_allocator.cc
    retry_allocator.cc
    memory_block.cc
    memory_block_desc.cc
    meta_cache.cc
    buddy_allocator.cc
    system_allocator.cc)

if(WITH_GPU OR WITH_ROCM)
  list(
    APPEND
    ALLOCATOR_SRCS
    cuda_allocator.cc
    cuda_managed_allocator.cc
    cuda_malloc_async_allocator.cc
    pinned_allocator.cc
    stream_safe_cuda_allocator.cc
    thread_local_allocator.cc)
endif()

if(CUDA_VERSION VERSION_GREATER_EQUAL 10.2)
  list(APPEND ALLOCATOR_SRCS cuda_virtual_mem_allocator.cc)
endif()

if(NOT WIN32)
  list(APPEND ALLOCATOR_SRCS mmap_allocator.cc)
  if(WITH_GPU)
    list(APPEND ALLOCATOR_SRCS cuda_ipc_allocator.cc)
  endif()
endif()

if(WITH_CUSTOM_DEVICE)
  list(APPEND ALLOCATOR_SRCS custom_allocator.cc
       stream_safe_custom_device_allocator.cc)
endif()

if(WITH_XPU)
  list(APPEND ALLOCATOR_SRCS xpu_allocator.cc stream_safe_xpu_allocator.cc)
endif()

collect_srcs(core_srcs SRCS ${ALLOCATOR_SRCS})
