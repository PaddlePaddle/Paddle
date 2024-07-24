nv_library(
  cuda_profiler
  SRCS cuda_profiler.cc
  DEPS enforce common)

nv_test(
  cudnn_helper_test
  SRCS cudnn_helper_test.cc
  DEPS phi common)
