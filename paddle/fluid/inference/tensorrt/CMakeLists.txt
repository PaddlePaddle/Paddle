nv_library(tensorrt_engine SRCS engine.cc DEPS framework_proto)
nv_test(test_tensorrt SRCS test_tensorrt.cc DEPS dynload_cuda device_context dynamic_loader)
nv_test(test_tensorrt_engine SRCS test_engine.cc DEPS dynload_cuda tensorrt_engine)
add_subdirectory(convert)
