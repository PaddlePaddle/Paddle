if (WITH_GPU)
     nv_test(gpu_timer_test SRCS gpu_timer_test.cu DEPS gtest)
elseif (WITH_ROCM)
    hip_test(gpu_timer_test SRCS gpu_timer_test.cu DEPS gtest)
endif()

cc_test(cache_test SRCS cache_test.cc DEPS gtest)
