cc_library(allocator SRCS allocator.cc DEPS place)
cc_library(cpu_allocator SRCS cpu_allocator.cc DEPS allocator)
cc_library(best_fit_allocator SRCS best_fit_allocator.cc DEPS allocator)
cc_library(locked_allocator SRCS locked_allocator.cc DEPS allocator)
nv_library(cuda_allocator SRCS cuda_allocator.cc DEPS allocator gpu_info)

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


cc_library(naive_managed_allocator SRCS naive_managed_allocator.cc DEPS allocator)
cc_test(naive_managed_allocator_test SRCS naive_managed_allocator_test.cc DEPS naive_managed_allocator)

if (WITH_GPU)
    set(AllocatorFacadeDeps gpu_info cuda_allocator)
else ()
    set(AllocatorFacadeDeps)
endif()

cc_library(aligned_allocator SRCS aligned_allocator.cc DEPS allocator)

cc_library(allocator_facade SRCS allocator_facade.cc DEPS
        ${AllocatorFacadeDeps}
        cpu_allocator
        locked_allocator
        best_fit_allocator
        naive_managed_allocator
        aligned_allocator)
