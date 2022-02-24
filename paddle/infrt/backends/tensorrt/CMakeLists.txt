cc_library(infrt_trt SRCS trt_engine.cc DEPS glog pten_dynload_cuda pten)

cc_test_tiny(test_infrt_trt SRCS test_trt_engine.cc DEPS infrt_trt pten_dynload_cuda tensorrt_converter)
