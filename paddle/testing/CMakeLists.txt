# for paddle test case

if(WITH_TESTING)
  set(paddle_gtest_main_deps
      device_context
      gtest
      gflags
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
endif()
