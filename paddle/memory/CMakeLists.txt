add_subdirectory(detail)

cc_library(memory SRCS memory.cc DEPS place enforce)
cc_library(memcpy SRCS memcpy.cc)

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

if(NOT WITH_C_API AND WITH_FLUID)
  file(GLOB MEMORY_HEADERS *.h)
  file(GLOB MEMORY_DETAIL_HEADERS detail/*.h)
  install(FILES ${MEMORY_HEADERS} DESTINATION include/paddle/memory)
  install(FILES ${MEMORY_DETAIL_HEADERS} DESTINATION include/paddle/memory/detail)
endif()
