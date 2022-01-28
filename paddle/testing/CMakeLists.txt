# for paddle test case

if(WITH_TESTING)
  cc_library(paddle_gtest_main SRCS paddle_gtest_main.cc DEPS init device_context memory gtest gflags)
endif()
cc_test(small_vector_test SRCS small_vector_test.cc DEPS gtest gflags)
