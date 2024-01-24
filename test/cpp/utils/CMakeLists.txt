if(NOT WITH_GFLAGS)
  paddle_test(flags_native_test SRCS flags_native_test.cc)
endif()

add_subdirectory(string)

paddle_test(array_ref_test SRCS array_ref_test.cc)
paddle_test(small_vector_test SRCS small_vector_test.cc)
paddle_test(variant_test SRCS variant_test.cc)
paddle_test(span_test SRCS span_test.cc)
