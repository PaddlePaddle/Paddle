cc_library(allocator SRCS allocator.cc DEPS place)
cc_library(cpu_allocator SRCS cpu_allocator.cc DEPS allocator)
cc_library(best_fit_allocator SRCS best_fit_allocator.cc DEPS allocator)
cc_library(locked_allocator SRCS locked_allocator.cc DEPS allocator)
cc_library(buffered_allocator SRCS buffered_allocator.cc DEPS allocator)
cc_library(legacy_allocator SRCS legacy_allocator.cc DEPS allocator buddy_allocator profiler)
cc_test(buffered_allocator_test SRCS buffered_allocator_test.cc DEPS best_fit_allocator locked_allocator buffered_allocator cpu_allocator)

if (WITH_GPU)
  nv_library(cuda_allocator SRCS cuda_allocator.cc DEPS allocator cuda_device_guard)
endif()

cc_library(retry_allocator SRCS retry_allocator.cc DEPS allocator)

if (WITH_GPU)
    nv_test(best_fit_allocator_test
            SRCS best_fit_allocator_test.cc
                 best_fit_allocator_test.cu
            DEPS best_fit_allocator
                 locked_allocator
                 cpu_allocator
                 cuda_allocator
                 device_context
                 memcpy)
else()
    cc_test(best_fit_allocator_test
            SRCS best_fit_allocator_test.cc
            DEPS best_fit_allocator
                 locked_allocator
                 cpu_allocator)
endif()

nv_library(pinned_allocator SRCS pinned_allocator.cc DEPS allocator)
if (WITH_GPU)
    set(AllocatorFacadeDeps gpu_info cuda_allocator pinned_allocator cuda_device_guard)
else ()
    set(AllocatorFacadeDeps)
endif()

cc_library(aligned_allocator SRCS aligned_allocator.cc DEPS allocator)
cc_library(auto_increment_allocator SRCS auto_increment_allocator.cc DEPS allocator)
cc_library(zero_size_allocator SRCS zero_size_allocator.cc DEPS allocator)
cc_library(conditional_allocator SRCS conditional_allocator.cc DEPS allocator)
cc_library(allocator_strategy SRCS allocator_strategy.cc DEPS gflags)
cc_library(allocator_facade SRCS allocator_facade.cc DEPS
        ${AllocatorFacadeDeps}
        cpu_allocator
        locked_allocator
        best_fit_allocator
        aligned_allocator
        auto_increment_allocator
        zero_size_allocator
        conditional_allocator
        retry_allocator
        buffered_allocator
        allocator_strategy
        legacy_allocator
        )

nv_test(allocation_and_eigen_test SRCS allocation_and_eigen_test.cu DEPS allocator_facade)

cc_test(retry_allocator_test SRCS retry_allocator_test.cc DEPS retry_allocator best_fit_allocator locked_allocator cpu_allocator)

cc_test(allocator_facade_test SRCS allocator_facade_test.cc DEPS allocator_facade)
