nv_library(
  cuda_profiler
  SRCS cuda_profiler.cc
  DEPS phi common)

nv_test(
  cudnn_helper_test
  SRCS cudnn_helper_test.cc
  DEPS phi common)
