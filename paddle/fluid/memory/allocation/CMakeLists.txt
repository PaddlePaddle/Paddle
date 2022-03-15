cc_library(allocator SRCS allocator.cc DEPS place)
cc_library(cpu_allocator SRCS cpu_allocator.cc DEPS allocator)
cc_library(locked_allocator SRCS locked_allocator.cc DEPS allocator)
cc_library(buffered_allocator SRCS buffered_allocator.cc DEPS allocator)
cc_library(best_fit_allocator SRCS best_fit_allocator.cc DEPS allocator)
cc_library(naive_best_fit_allocator SRCS naive_best_fit_allocator.cc DEPS allocator buddy_allocator profiler)
cc_test(naive_best_fit_allocator_test SRCS naive_best_fit_allocator_test.cc DEPS naive_best_fit_allocator)
cc_test(buffered_allocator_test SRCS buffered_allocator_test.cc DEPS locked_allocator buffered_allocator cpu_allocator best_fit_allocator)

if (WITH_MKLDNN)
  set(MKLDNN_CTX_DEPS mkldnn)
else ()
  set(MKLDNN_CTX_DEPS)
endif()

if (WITH_GPU)
  nv_library(cuda_allocator SRCS cuda_allocator.cc DEPS allocator cuda_device_guard)
  nv_library(cuda_managed_allocator SRCS cuda_managed_allocator.cc DEPS allocator cuda_device_guard gpu_info)
  nv_library(pinned_allocator SRCS pinned_allocator.cc DEPS allocator)
  nv_library(stream_safe_cuda_allocator SRCS stream_safe_cuda_allocator.cc DEPS allocator cuda_graph)
  nv_library(thread_local_allocator SRCS thread_local_allocator.cc DEPS allocator)

  cc_test(thread_local_allocator_test SRCS thread_local_allocator_test.cc DEPS thread_local_allocator)
  if(CUDA_VERSION GREATER_EQUAL 10.2)
    nv_library(cuda_virtual_mem_allocator SRCS cuda_virtual_mem_allocator.cc DEPS dynload_cuda)
  endif()
endif()

if (WITH_ROCM)
  hip_library(cuda_allocator SRCS cuda_allocator.cc DEPS allocator cuda_device_guard)
  hip_library(cuda_managed_allocator SRCS cuda_managed_allocator.cc DEPS allocator cuda_device_guard gpu_info)
  hip_library(pinned_allocator SRCS pinned_allocator.cc DEPS allocator)
  hip_library(stream_safe_cuda_allocator SRCS stream_safe_cuda_allocator.cc DEPS allocator)
  hip_library(thread_local_allocator SRCS thread_local_allocator.cc DEPS allocator)
  
  cc_test(thread_local_allocator_test SRCS thread_local_allocator_test.cc DEPS thread_local_allocator)
endif()

if (WITH_ASCEND_CL)
  cc_library(npu_allocator SRCS npu_allocator.cc DEPS allocator npu_info)
  cc_library(npu_pinned_allocator SRCS npu_pinned_allocator.cc DEPS allocator npu_info)
endif()

cc_library(retry_allocator SRCS retry_allocator.cc DEPS allocator)

if (WITH_GPU OR WITH_ROCM)
    set(AllocatorFacadeDeps gpu_info cuda_allocator cuda_managed_allocator pinned_allocator cuda_device_guard thread_local_allocator stream_safe_cuda_allocator device_context)
    if(CUDA_VERSION GREATER_EQUAL 10.2)
      list(APPEND AllocatorFacadeDeps cuda_virtual_mem_allocator)
    endif()
elseif(WITH_XPU)
    set(AllocatorFacadeDeps xpu_info)
elseif(WITH_IPU)
    set(AllocatorFacadeDeps ipu_info)
elseif(WITH_ASCEND)
    set(AllocatorFacadeDeps ascend_npu_info)
else ()
    set(AllocatorFacadeDeps)
endif()

if (WITH_CUSTOM_DEVICE)
  cc_library(custom_allocator SRCS custom_allocator.cc DEPS allocator device_manager)
  set(AllocatorFacadeDeps ${AllocatorFacadeDeps} custom_allocator)
endif()

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
elseif (WITH_ROCM)
    hip_test(best_fit_allocator_test
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

list(APPEND AllocatorFacadeDeps cpu_allocator locked_allocator aligned_allocator retry_allocator buffered_allocator naive_best_fit_allocator auto_growth_best_fit_allocator virtual_memory_auto_growth_best_fit_allocator best_fit_allocator)

if (WITH_ASCEND_CL)
    list(APPEND AllocatorFacadeDeps npu_pinned_allocator)
endif()


cc_library(aligned_allocator SRCS aligned_allocator.cc DEPS allocator)
cc_test(test_aligned_allocator SRCS test_aligned_allocator.cc DEPS aligned_allocator)
cc_library(allocator_strategy SRCS allocator_strategy.cc DEPS gflags ${AllocatorFacadeDeps})
cc_library(allocator_facade SRCS allocator_facade.cc DEPS allocator_strategy)

if (WITH_GPU)
  target_link_libraries(allocator_facade cuda_graph)
endif()

cc_test(retry_allocator_test SRCS retry_allocator_test.cc DEPS retry_allocator locked_allocator cpu_allocator)
if (WITH_TESTING)
  if ((WITH_GPU OR WITH_ROCM) AND TARGET retry_allocator_test)
    target_link_libraries(retry_allocator_test cuda_allocator)
  endif()

  if (TEST retry_allocator_test)
    set_tests_properties(retry_allocator_test PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  endif()
endif()

cc_test(allocator_facade_abs_flags_test SRCS allocator_facade_abs_flags_test.cc DEPS allocator_facade)

cc_test(allocator_facade_frac_flags_test SRCS allocator_facade_frac_flags_test.cc DEPS allocator_facade)

cc_library(auto_growth_best_fit_allocator SRCS auto_growth_best_fit_allocator.cc DEPS allocator aligned_allocator flags)
cc_test(auto_growth_best_fit_allocator_facade_test SRCS auto_growth_best_fit_allocator_facade_test.cc DEPS cpu_allocator auto_growth_best_fit_allocator)
cc_test(auto_growth_best_fit_allocator_test SRCS auto_growth_best_fit_allocator_test.cc DEPS auto_growth_best_fit_allocator)

cc_library(virtual_memory_auto_growth_best_fit_allocator SRCS virtual_memory_auto_growth_best_fit_allocator.cc DEPS allocator aligned_allocator)

if(NOT WIN32)
  cc_library(mmap_allocator SRCS mmap_allocator.cc DEPS allocator)
  cc_test(mmap_allocator_test SRCS mmap_allocator_test.cc DEPS mmap_allocator allocator)
  if (WITH_GPU)
    cc_library(cuda_ipc_allocator SRCS cuda_ipc_allocator.cc DEPS allocator)
  endif()
endif(NOT WIN32)
