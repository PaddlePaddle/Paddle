add_subdirectory(detail)

cc_library(memory
    SRCS
    memory.cc)

cc_library(paddle_memory
    DEPS
    memory meta_data
    meta_cache memory_block
    buddy_allocator system_allocator)
