# for paddle test case

if(WITH_TESTING)
  set(paddle_gtest_main_deps
      device_context
      gtest
      phi
      common
      init
      phi_utils
      proto_desc)

  cc_library(
    paddle_gtest_main
    SRCS paddle_gtest_main.cc
    DEPS ${paddle_gtest_main_deps})

  if(LINUX AND (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 8.2))
    cc_library(
      paddle_gtest_main_new
      SRCS paddle_gtest_main.cc
      DEPS gtest
           xxhash
           eigen3
           dlpack
           common
           init
           phi
           phi_utils)
  else()
    cc_library(
      paddle_gtest_main_new
      SRCS paddle_gtest_main.cc
      DEPS gtest xxhash framework_proto eigen3 dlpack)
  endif()

  if(WITH_ONEDNN)
    add_dependencies(paddle_gtest_main_new onednn)
  endif()
endif()
