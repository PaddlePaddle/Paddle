# for paddle test case

if(WITH_TESTING)
  add_library(paddle_test_main STATIC TestMain.cpp)
  add_dependencies(paddle_test_main paddle_proto ${external_project_dependencies})
  add_library(paddle_test_util STATIC TestUtil.cpp)
  add_dependencies(paddle_test_util paddle_proto ${external_project_dependencies})
  if(NOT MOBILE_INFERENCE)
    cc_library(paddle_gtest_main SRCS paddle_gtest_main.cc DEPS init memory gtest gflags)
  endif()
endif()
