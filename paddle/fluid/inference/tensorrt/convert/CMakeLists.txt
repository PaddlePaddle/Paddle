nv_library(tensorrt_convert SRCS convert.cc DEPS dynload_cuda)
nv_test(tensorrt_convert_test SRCS convert_test.cc DEPS tensorrt paddle_fluid)
