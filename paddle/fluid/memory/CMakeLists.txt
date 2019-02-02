# make the external project built first
set(PADDLE_MEMORY_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/memory/build")
add_custom_command(OUTPUT ${PADDLE_MEMORY_BUILD_DIR}/.timestamp
        COMMAND ${CMAKE_COMMAND} -E touch ${PADDLE_MEMORY_BUILD_DIR}/.timestamp
        DEPENDS ${external_project_dependencies})

add_subdirectory(detail)
add_subdirectory(allocation)
cc_library(malloc SRCS malloc.cc DEPS place enforce allocator_facade)
cc_library(memcpy SRCS memcpy.cc DEPS place)

cc_library(memory
        DEPS
        malloc
        memcpy)
#if (WITH_GPU)
#   nv_test(pinned_memory_test SRCS pinned_memory_test.cu  DEPS place memory)
#endif()
