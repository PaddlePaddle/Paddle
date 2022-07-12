include(ExternalProject)

set(ALLOCATION_DEPS place stats profiler)
set(ALLOCATOR_SRCS
    allocator.cc
    cpu_allocator.cc
    locked_allocator.cc
    aligned_allocator.cc
    buffered_allocator.cc
    best_fit_allocator.cc
    naive_best_fit_allocator.cc
    allocator_strategy.cc
    allocator_facade.cc
    auto_growth_best_fit_allocator.cc
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
    pinned_allocator.cc
    stream_safe_cuda_allocator.cc
    thread_local_allocator.cc)
  list(APPEND ALLOCATOR_DEPS cuda_device_guard gpu_info cuda_graph dynload_cuda)
endif()

if(CUDA_VERSION GREATER_EQUAL 10.2)
  list(APPEND ALLOCATOR_SRCS cuda_virtual_mem_allocator.cc)
endif()

if(NOT WIN32)
  list(APPEND ALLOCATOR_SRCS mmap_allocator.cc)
  if(WITH_GPU)
    list(APPEND ALLOCATOR_SRCS cuda_ipc_allocator.cc)
  endif()
endif()

if(WITH_ASCEND_CL)
  list(APPEND ALLOCATOR_SRCS npu_allocator.cc npu_pinned_allocator.cc)
  list(APPEND ALLOCATOR_DEPS npu_info)
endif()

if(WITH_CUSTOM_DEVICE)
  list(APPEND ALLOCATOR_SRCS custom_allocator.cc)
  list(APPEND ALLOCATOR_DEPS device_manager)
endif()

if(WITH_XPU)
  list(APPEND ALLOCATOR_DEPS xpu_info)
endif()

if(WITH_IPU)
  list(APPEND ALLOCATOR_DEPS ipu_info)
endif()

add_library(allocator "${ALLOCATOR_SRCS}")
target_link_libraries(allocator ${ALLOCATOR_DEPS})

cc_test(
  naive_best_fit_allocator_test
  SRCS naive_best_fit_allocator_test.cc
  DEPS allocator)
cc_test(
  buffered_allocator_test
  SRCS buffered_allocator_test.cc
  DEPS allocator)

if(WITH_GPU)
  nv_test(
    thread_local_allocator_test
    SRCS thread_local_allocator_test.cc
    DEPS allocator)
endif()
if(WITH_ROCM)
  hip_test(
    thread_local_allocator_test
    SRCS thread_local_allocator_test.cc
    DEPS allocator)
endif()

if(WITH_GPU)
  nv_test(
    best_fit_allocator_test
    SRCS best_fit_allocator_test.cc best_fit_allocator_test.cu
    DEPS allocator memcpy)
elseif(WITH_ROCM)
  hip_test(
    best_fit_allocator_test
    SRCS best_fit_allocator_test.cc best_fit_allocator_test.cu
    DEPS allocator memcpy)
else()
  cc_test(
    best_fit_allocator_test
    SRCS best_fit_allocator_test.cc
    DEPS allocator)
endif()

cc_test(
  test_aligned_allocator
  SRCS test_aligned_allocator.cc
  DEPS allocator)

cc_test(
  retry_allocator_test
  SRCS retry_allocator_test.cc
  DEPS allocator)
if(TEST retry_allocator_test)
  set_tests_properties(retry_allocator_test PROPERTIES LABELS
                                                       "RUN_TYPE=EXCLUSIVE")
endif()

cc_test(
  allocator_facade_abs_flags_test
  SRCS allocator_facade_abs_flags_test.cc
  DEPS allocator)

cc_test(
  allocator_facade_frac_flags_test
  SRCS allocator_facade_frac_flags_test.cc
  DEPS allocator)

cc_test(
  auto_growth_best_fit_allocator_facade_test
  SRCS auto_growth_best_fit_allocator_facade_test.cc
  DEPS allocator)
cc_test(
  auto_growth_best_fit_allocator_test
  SRCS auto_growth_best_fit_allocator_test.cc
  DEPS allocator)

if(NOT WIN32)
  cc_test(
    mmap_allocator_test
    SRCS mmap_allocator_test.cc
    DEPS allocator)
endif()

cc_test(
  system_allocator_test
  SRCS system_allocator_test.cc
  DEPS allocator)

cc_test(
  buddy_allocator_test
  SRCS buddy_allocator_test.cc
  DEPS allocator)

function(file_download_and_uncompress URL NAME)
  message(STATUS "Download dependence[${NAME}] from ${URL}")
  set(${NAME}_INCLUDE_DIR
      ${THIRD_PARTY_PATH}/${NAME}
      PARENT_SCOPE)
  ExternalProject_Add(
    extern_download_${NAME}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX ${THIRD_PARTY_PATH}/${NAME}
    URL ${URL}
    DOWNLOAD_DIR ${THIRD_PARTY_PATH}/${NAME}
    SOURCE_DIR ${THIRD_PARTY_PATH}/${NAME}
    DOWNLOAD_NO_PROGRESS 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND "")
  set(third_party_deps
      ${third_party_deps} extern_download_${NAME}
      PARENT_SCOPE)
endfunction()

if(WITH_TESTING)
  if(TEST buddy_allocator_test)
    set_tests_properties(buddy_allocator_test PROPERTIES LABELS
                                                         "RUN_TYPE=EXCLUSIVE")
  endif()
  set(URL "https://paddle-ci.cdn.bcebos.com/buddy_allocator_test_data.tar")
  file_download_and_uncompress(URL "buddy_allocator")
endif()
