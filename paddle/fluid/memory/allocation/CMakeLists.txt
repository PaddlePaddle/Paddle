include(ExternalProject)

set(ALLOCATOR_DEPS profiler phi common device_context)
set(ALLOCATOR_SRCS
    allocator.cc
    cpu_allocator.cc
    aligned_allocator.cc
    buffered_allocator.cc
    best_fit_allocator.cc
    naive_best_fit_allocator.cc
    allocator_strategy.cc
    allocator_facade.cc
    auto_growth_best_fit_allocator.cc
    auto_growth_best_fit_allocator_v2.cc
    virtual_memory_auto_growth_best_fit_allocator.cc
    retry_allocator.cc
    memory_block.cc
    memory_block_desc.cc
    meta_cache.cc
    buddy_allocator.cc
    system_allocator.cc)

if(WITH_GPU OR WITH_ROCM)
  list(
    APPEND
    ALLOCATOR_SRCS
    cuda_allocator.cc
    cuda_managed_allocator.cc
    cuda_malloc_async_allocator.cc
    pinned_allocator.cc
    stream_safe_cuda_allocator.cc
    thread_local_allocator.cc)
  list(APPEND ALLOCATOR_DEPS gpu_info phi common)
endif()

if(WITH_ROCM)
  list(APPEND ALLOCATOR_DEPS ${ROCM_HIPRTC_LIB})
endif()

if(WITH_GPU OR WITH_ROCM)
  list(APPEND ALLOCATOR_DEPS phi common)
endif()
if(CUDA_VERSION VERSION_GREATER_EQUAL 10.2)
  list(APPEND ALLOCATOR_SRCS cuda_virtual_mem_allocator.cc)
endif()

if(NOT WIN32)
  list(APPEND ALLOCATOR_SRCS mmap_allocator.cc)
  if(WITH_GPU)
    list(APPEND ALLOCATOR_SRCS cuda_ipc_allocator.cc)
  endif()
endif()

if(UNIX AND NOT APPLE)
  list(APPEND ALLOCATOR_DEPS rt)
endif()

if(WITH_CUSTOM_DEVICE)
  list(APPEND ALLOCATOR_SRCS custom_allocator.cc
       stream_safe_custom_device_allocator.cc)
endif()

if(WITH_XPU)
  list(APPEND ALLOCATOR_DEPS xpu_info)
  list(APPEND ALLOCATOR_SRCS xpu_allocator.cc stream_safe_xpu_allocator.cc)
endif()

if(WITH_IPU)
  list(APPEND ALLOCATOR_DEPS ipu_info)
endif()

add_library(allocator "${ALLOCATOR_SRCS}")
target_link_libraries(allocator ${ALLOCATOR_DEPS})
# note: why only add dependency for framework_proto.
# Because it is needed to generate framework.pb.h used in some header files.
add_dependencies(allocator framework_proto)
set_property(GLOBAL PROPERTY FLUID_MODULES allocator)

if(WITH_TESTING)
  # TODO(zhiqiu): why not win32? because wget is not found on windows
  if(NOT WIN32)
    add_custom_target(
      download_data
      COMMAND wget -nc --no-check-certificate
              https://paddle-ci.cdn.bcebos.com/buddy_allocator_test_data.tar
      COMMAND tar -xf buddy_allocator_test_data.tar)
  endif()
endif()
