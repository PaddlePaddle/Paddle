# for paddle test case

if(WITH_TESTING)
  set(paddle_gtest_main_deps
      device_context
      gtest
      phi
      common
      init
      memory
      phi_utils
      proto_desc)

  if(WITH_GPU OR WITH_ROCM)
    list(APPEND paddle_gtest_main_deps gpu_info)
  endif()

  cc_library(
    paddle_gtest_main
    SRCS paddle_gtest_main.cc
    DEPS ${paddle_gtest_main_deps})

  cc_library(
    paddle_gtest_main_new
    SRCS paddle_gtest_main.cc
    DEPS gtest xxhash framework_proto eigen3 dlpack)
  if(WITH_MKLDNN)
    add_dependencies(paddle_gtest_main_new mkldnn)
  endif()
endif()
