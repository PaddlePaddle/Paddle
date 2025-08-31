if(NOT WIN32)
  if(WITH_GPU)
    paddle_test(compat_basic_test SRCS compat_basic_test.cc)
    paddle_test(torch_library_test SRCS torch_library_test.cc)
    target_link_libraries(compat_basic_test ${CUDA_LIBRARIES}
                          ${CUDA_CUDART_LIBRARY})
  endif()
endif()
