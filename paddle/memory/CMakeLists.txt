add_subdirectory(detail)

cc_library(memory SRCS memory.cc DEPS place enforce)
cc_library(memcpy SRCS memcpy.cc DEPS place)

cc_library(paddle_memory
    DEPS
    memory
    memcpy
    meta_data
    meta_cache
    memory_block
    buddy_allocator
    system_allocator)

cc_test(memory_test SRCS memory_test.cc DEPS place paddle_memory)
