add_subdirectory(allocation)

file(GLOB fluid_memory_srcs "*.cc")

if(WITH_MKLDNN)
  set(MKLDNN_CTX_DEPS mkldnn)
else()
  set(MKLDNN_CTX_DEPS)
endif()
set(fluid_memory_deps place enforce common allocator ${MKLDNN_CTX_DEPS})

cc_library(
  fluid_memory
  SRCS ${fluid_memory_srcs}
  DEPS ${fluid_memory_deps})
