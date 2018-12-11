# for paddle test case

if(WITH_TESTING)
  add_library(paddle_test_main STATIC TestMain.cpp)
  add_dependencies(paddle_test_main paddle_proto ${external_project_dependencies})
  if(NOT WIN32)
    add_library(paddle_test_util STATIC TestUtil.cpp)
    add_dependencies(paddle_test_util paddle_proto ${external_project_dependencies})
  endif(NOT WIN32)
  if(NOT MOBILE_INFERENCE)
    cc_library(paddle_gtest_main SRCS paddle_gtest_main.cc DEPS device_context memory gtest gflags)
  endif()
endif()
