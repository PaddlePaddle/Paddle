nv_test(test_tensorrt SRCS test_tensorrt.cc DEPS dynload_cuda device_context dynamic_loader)
cc_library(tensorrt DEPS tensorrt_convert)
add_subdirectory(convert)
