add_subdirectory(detail)

if(WITH_FPGA)
    cc_library(memory SRCS memory.cc DEPS polaris)
    cc_library(memcpy SRCS memcpy.cc DEPS polaris)
else()
    cc_library(memory SRCS memory.cc)
    cc_library(memcpy SRCS memcpy.cc)
endif()

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
