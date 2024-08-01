add_subdirectory(allocation)

file(GLOB fluid_memory_srcs "*.cc")

if(WITH_ONEDNN)
  set(ONEDNN_CTX_DEPS onednn)
else()
  set(ONEDNN_CTX_DEPS)
endif()
set(fluid_memory_deps phi common allocator ${ONEDNN_CTX_DEPS})

cc_library(
  fluid_memory
  SRCS ${fluid_memory_srcs}
  DEPS ${fluid_memory_deps})
