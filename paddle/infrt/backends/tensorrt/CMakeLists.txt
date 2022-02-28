cc_library(infrt_trt SRCS trt_engine.cc DEPS glog phi_dynload_cuda phi)

cc_test_tiny(test_infrt_trt SRCS test_trt_engine.cc DEPS infrt_trt phi_dynload_cuda tensorrt_converter)
