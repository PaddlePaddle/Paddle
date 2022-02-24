add_subdirectory(string)
cc_test(small_vector_test SRCS small_vector_test.cc DEPS gtest gflags)
cc_test(array_ref_test SRCS array_ref_test.cc DEPS gtest gflags)
