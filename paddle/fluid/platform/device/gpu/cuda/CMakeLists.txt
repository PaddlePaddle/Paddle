nv_library(
<<<<<<< HEAD
=======
  cuda_graph
  SRCS cuda_graph.cc
  DEPS enforce)
nv_library(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  cuda_profiler
  SRCS cuda_profiler.cc
  DEPS enforce)

nv_test(
  cudnn_helper_test
  SRCS cudnn_helper_test.cc
  DEPS dynload_cuda)
