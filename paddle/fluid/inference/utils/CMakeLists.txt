cc_library(
  benchmark
  SRCS benchmark.cc
  DEPS enforce)
cc_test(
  test_benchmark
  SRCS benchmark_tester.cc
  DEPS benchmark)
cc_library(
  infer_io_utils
  SRCS io_utils.cc
  DEPS paddle_inference_api lod_tensor shape_range_info_proto)
cc_library(
  model_utils
  SRCS model_utils.cc
  DEPS proto_desc enforce)
cc_test(
  infer_io_utils_tester
  SRCS io_utils_tester.cc
  DEPS infer_io_utils)
cc_library(table_printer SRCS table_printer.cc)
cc_test(
  test_table_printer
  SRCS table_printer_tester.cc
  DEPS table_printer)

proto_library(shape_range_info_proto SRCS shape_range_info.proto)
