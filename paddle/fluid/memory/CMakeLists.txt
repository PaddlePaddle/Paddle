add_subdirectory(detail)
add_subdirectory(allocation)
cc_library(malloc SRCS malloc.cc DEPS place enforce allocator_facade profiler)
cc_library(memcpy SRCS memcpy.cc DEPS place)

cc_library(memory
        DEPS
        malloc
        memcpy)
#if (WITH_GPU)
#   nv_test(pinned_memory_test SRCS pinned_memory_test.cu  DEPS place memory)
#endif()
