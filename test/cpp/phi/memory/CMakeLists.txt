if(WITH_GPU)
  if(WIN32)
    message(STATUS "Skip compact_allocator_test on Windows")
  else()
    nv_test(
      compact_allocator_test
      SRCS compact_allocator_test.cc
      DEPS phi common)
  endif()
endif()

if(WITH_GPU)
  if(WIN32)
    message(STATUS "Skip gen_compact_test on Windows")
  else()
    nv_test(
      gen_compact_test
      SRCS gen_compact_test.cc
      DEPS phi common)
  endif()
endif()
