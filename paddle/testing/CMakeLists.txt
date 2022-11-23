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
<<<<<<< HEAD
=======

  cc_library(
    paddle_gtest_main_new
    SRCS paddle_gtest_main.cc
    DEPS gtest xxhash framework_proto eigen3 dlpack)
  if(WITH_MKLDNN)
    add_dependencies(paddle_gtest_main_new mkldnn)
  endif()
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
endif()
