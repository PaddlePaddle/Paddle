add_subdirectory(detail)
add_subdirectory(allocation)

set(AllocatorDeps cpu_allocator
        locked_allocator
        best_fit_allocator
        aligned_allocator
        auto_increment_allocator
        zero_size_allocator
        conditional_allocator
        retry_allocator
        buffered_allocator
        allocator_strategy
        legacy_allocator ${AllocatorFacadeDeps})

if (WITH_GPU)
    list(APPEND AllocatorDeps gpu_info cuda_allocator pinned_allocator cuda_device_guard)
endif()

cc_library(malloc SRCS malloc.cc DEPS ${AllocatorDeps})
cc_library(memcpy SRCS memcpy.cc DEPS ${AllocatorDeps})

cc_library(memory
        DEPS
        malloc
        memcpy)

cc_test(allocator_test SRCS allocator_test.cc DEPS memory)

#if (WITH_GPU)
#   nv_test(pinned_memory_test SRCS pinned_memory_test.cu  DEPS place memory)
#endif()
