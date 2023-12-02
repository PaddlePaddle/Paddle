cc_test(
  naive_best_fit_allocator_test
  SRCS naive_best_fit_allocator_test.cc
  DEPS allocator)
paddle_test(buffered_allocator_test SRCS buffered_allocator_test.cc DEPS
            allocator)

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
