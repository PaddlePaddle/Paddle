add_subdirectory(detail)

cc_library(memory SRCS memory.cc DEPS place enforce)
cc_library(memcpy SRCS memcpy.cc DEPS place)

cc_library(paddle_memory
        DEPS
        memory
        memcpy
        buddy_allocator)

cc_test(memory_test SRCS memory_test.cc DEPS place paddle_memory)

#if (WITH_GPU)
#   nv_test(pinned_memory_test SRCS pinned_memory_test.cu  DEPS place paddle_memory)
#endif()
