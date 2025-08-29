paddle_test(compat_basic_test SRCS compat_basic_test.cc)
paddle_test(torch_library_test SRCS torch_library_test.cc)

if(WITH_CUDA)
  target_link_libraries(compat_basic_test ${CUDA_LIBRARIES})
endif()
