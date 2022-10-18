nv_library(
  cuda_graph
  SRCS cuda_graph.cc
  DEPS enforce)
nv_library(
  cuda_profiler
  SRCS cuda_profiler.cc
  DEPS enforce)
if(WITH_CUDNN_FRONTEND)
  nv_library(
    cudnn_frontend
    SRCS cudnn_frontend.cc
    DEPS gflags)
endif()
nv_test(
  cudnn_helper_test
  SRCS cudnn_helper_test.cc
  DEPS dynload_cuda)
