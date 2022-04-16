add_subdirectory(plugin)

core_gather_headers()

gather_srcs(infrt_src SRCS trt_engine.cc)

cc_test_tiny(test_infrt_trt SRCS test_trt_engine.cc DEPS infrt phi_dynload_cuda tensorrt_converter)
