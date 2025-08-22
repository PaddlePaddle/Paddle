paddle_test(conversion_basic_test SRCS conversion_basic_test.cc)
paddle_test(torch_library_test SRCS torch_library_test.cc)

target_link_libraries(conversion_basic_test ${CUDA_LIBRARIES})
