# for paddle test case

if(WITH_TESTING)
  cc_library(paddle_gtest_main SRCS paddle_gtest_main.cc DEPS init device_context memory gtest gflags proto_desc phi_utils)
endif()
