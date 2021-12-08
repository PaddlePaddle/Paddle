hip_library(rocm_info SRCS rocm_info.cc DEPS gflags glog enforce monitor dynload_cuda)

hip_test(miopen_helper_test SRCS miopen_helper_test.cc DEPS dynload_cuda)
