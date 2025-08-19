paddle_test(conversion_basic_test SRCS conversion_basic_test.cc)

target_link_libraries(conversion_basic_test ${CUDA_LIBRARIES})
