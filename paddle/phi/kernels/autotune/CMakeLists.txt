<<<<<<< HEAD
if(WITH_GPU)
  nv_test(
    gpu_timer_test
    SRCS gpu_timer_test.cu
    DEPS gtest)
  nv_test(
    auto_tune_test
    SRCS auto_tune_test.cu
    DEPS gtest)
elseif(WITH_ROCM)
  hip_test(
    gpu_timer_test
    SRCS gpu_timer_test.cu
    DEPS gtest)
  hip_test(
    auto_tune_test
    SRCS auto_tune_test.cu
    DEPS gtest)
endif()

cc_library(cache SRCS cache.cc)
=======
if(WITH_CUDNN_FRONTEND)
  cc_library(
    cache
    SRCS cache.cc
    DEPS cudnn-frontend)
else()
  cc_library(cache SRCS cache.cc)
endif()
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
cc_library(
  switch_autotune
  SRCS switch_autotune.cc
  DEPS cache flags)
<<<<<<< HEAD

cc_test(
  cache_test
  SRCS cache_test.cc
  DEPS gtest cache)
=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
