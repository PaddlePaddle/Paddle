nv_library(tensorrt_convert SRCS convert.cc mul_op.cc conv2d_op.cc DEPS dynload_cuda)
nv_test(test_tensorrt_convert SRCS test_convert.cc DEPS tensorrt paddle_fluid)
