add_subdirectory(detail)
add_subdirectory(allocation)

if (WITH_MKLDNN)
    set(MKLDNN_CTX_DEPS mkldnn)
else ()
    set(MKLDNN_CTX_DEPS)
endif()

cc_library(malloc SRCS malloc.cc DEPS
    place enforce allocator_facade profiler ${MKLDNN_CTX_DEPS})
cc_library(memcpy SRCS memcpy.cc DEPS place device_context)
cc_library(stats SRCS stats.cc DEPS enforce)
cc_library(memory DEPS malloc memcpy stats)

cc_test(stats_test SRCS stats_test.cc DEPS stats)

if (WITH_GPU)
    nv_test(malloc_test
            SRCS malloc_test.cu
            DEPS device_context malloc)
    nv_test(stream_safe_cuda_alloc_test
            SRCS stream_safe_cuda_alloc_test.cu
            DEPS malloc cuda_graph_with_memory_pool)
    nv_test(cuda_managed_memory_test
            SRCS cuda_managed_memory_test.cu
            DEPS malloc gpu_info place)
    
    if(WITH_TESTING AND TEST stream_safe_cuda_alloc_test)
        set_tests_properties(stream_safe_cuda_alloc_test PROPERTIES 
                             ENVIRONMENT "FLAGS_use_stream_safe_cuda_allocator=true;FLAGS_allocator_strategy=auto_growth")
    endif()
endif()

if (WITH_ROCM)
    hip_test(malloc_test
            SRCS malloc_test.cu
            DEPS device_context malloc)
    hip_test(cuda_managed_memory_test
            SRCS cuda_managed_memory_test.cu
            DEPS malloc gpu_info place)
endif()

if(WITH_TESTING AND TEST cuda_managed_memory_test)
set_tests_properties(cuda_managed_memory_test PROPERTIES
                     ENVIRONMENT "FLAGS_use_cuda_managed_memory=true;FLAGS_allocator_strategy=auto_growth"
                     TIMEOUT 50)
endif()

if(WITH_GPU AND WITH_TESTING AND NOT "$ENV{CI_SKIP_CPP_TEST}" STREQUAL "ON")       
  nv_test(get_base_ptr_test SRCS get_base_ptr_test.cu DEPS malloc gpu_info)
  set_tests_properties(get_base_ptr_test PROPERTIES 
                       ENVIRONMENT "FLAGS_allocator_strategy=auto_growth;
                                    FLAGS_use_stream_safe_cuda_allocator=true;")
endif()

#if (WITH_GPU)
#   nv_test(pinned_memory_test SRCS pinned_memory_test.cu  DEPS place memory)
#endif()
