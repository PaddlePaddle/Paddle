cc_library(benchmark SRCS benchmark.cc DEPS enforce)
cc_test(test_benchmark SRCS benchmark_tester.cc DEPS benchmark)
cc_library(infer_io_utils SRCS io_utils.cc DEPS paddle_inference_api lod_tensor)
cc_test(infer_io_utils_tester SRCS io_utils_tester.cc DEPS infer_io_utils)
