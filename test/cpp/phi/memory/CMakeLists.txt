if(WITH_GPU)
  if(WIN32)
    message(STATUS "Skip compact_allocator_test on Windows")
  else()
    paddle_test(compact_allocator_test SRCS compact_allocator_test.cc DEPS phi
                common)
  endif()
endif()
