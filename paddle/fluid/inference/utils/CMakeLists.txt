cc_library(
  benchmark
  SRCS benchmark.cc
  DEPS enforce common)
paddle_test(test_benchmark SRCS benchmark_tester.cc DEPS benchmark)
cc_library(
  infer_io_utils
  SRCS io_utils.cc
  DEPS paddle_inference_api lod_tensor shape_range_info_proto)
cc_library(
  model_utils
  SRCS model_utils.cc
  DEPS proto_desc enforce common)

cc_library(table_printer SRCS table_printer.cc)
paddle_test(test_table_printer SRCS table_printer_tester.cc)

proto_library(shape_range_info_proto SRCS shape_range_info.proto)
