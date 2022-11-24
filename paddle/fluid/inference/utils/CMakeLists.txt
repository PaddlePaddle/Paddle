cc_library(
  benchmark
  SRCS benchmark.cc
  DEPS enforce)
<<<<<<< HEAD
cc_test(
  test_benchmark
  SRCS benchmark_tester.cc
  DEPS benchmark)
=======
cc_test_old(test_benchmark SRCS benchmark_tester.cc DEPS benchmark)
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
cc_library(
  infer_io_utils
  SRCS io_utils.cc
  DEPS paddle_inference_api lod_tensor shape_range_info_proto)
cc_library(
  model_utils
  SRCS model_utils.cc
  DEPS proto_desc enforce)
<<<<<<< HEAD
cc_test(
  infer_io_utils_tester
  SRCS io_utils_tester.cc
  DEPS infer_io_utils)
=======
cc_test_old(infer_io_utils_tester SRCS io_utils_tester.cc DEPS infer_io_utils)
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(infer_io_utils_tester)
endif()

cc_library(table_printer SRCS table_printer.cc)
<<<<<<< HEAD
cc_test(
  test_table_printer
  SRCS table_printer_tester.cc
  DEPS table_printer)
=======
cc_test_old(test_table_printer SRCS table_printer_tester.cc DEPS table_printer)
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

proto_library(shape_range_info_proto SRCS shape_range_info.proto)
