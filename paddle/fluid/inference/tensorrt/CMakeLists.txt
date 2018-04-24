if(WITH_TESTING)
  nv_test(test_tensorrt SRCS test_tensorrt.cc DEPS dynload_cuda device_context dynamic_loader)
  nv_test(test_tensorrt_engine SRCS test_engine.cc engine.cc DEPS dynload_cuda)
endif()
