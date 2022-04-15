if (WITH_GPU)
  nv_test(gpu_timer_test SRCS gpu_timer_test.cu DEPS gtest)
  nv_test(auto_tune_test SRCS auto_tune_test.cu DEPS gtest)
elseif (WITH_ROCM)
  hip_test(gpu_timer_test SRCS gpu_timer_test.cu DEPS gtest)
  hip_test(auto_tune_test SRCS auto_tune_test.cu DEPS gtest)
endif()

cc_library(cache SRCS cache.cc DEPS boost)
cc_library(switch_autotune SRCS switch_autotune.cc DEPS cache flags)

cc_test(cache_test SRCS cache_test.cc DEPS gtest cache)
