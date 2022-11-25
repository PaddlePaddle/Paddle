cc_library(
  benchmark
  SRCS benchmark.cc
  DEPS enforce)
<<<<<<< HEAD
cc_test_old(test_benchmark SRCS benchmark_tester.cc DEPS benchmark)
=======
cc_test(
  test_benchmark
  SRCS benchmark_tester.cc
  DEPS benchmark)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
cc_library(
  infer_io_utils
  SRCS io_utils.cc
  DEPS paddle_inference_api lod_tensor shape_range_info_proto)
cc_library(
  model_utils
  SRCS model_utils.cc
  DEPS proto_desc enforce)
<<<<<<< HEAD
cc_test_old(infer_io_utils_tester SRCS io_utils_tester.cc DEPS infer_io_utils)
=======
cc_test(
  infer_io_utils_tester
  SRCS io_utils_tester.cc
  DEPS infer_io_utils)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(infer_io_utils_tester)
endif()

cc_library(table_printer SRCS table_printer.cc)
<<<<<<< HEAD
cc_test_old(test_table_printer SRCS table_printer_tester.cc DEPS table_printer)
=======
cc_test(
  test_table_printer
  SRCS table_printer_tester.cc
  DEPS table_printer)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

proto_library(shape_range_info_proto SRCS shape_range_info.proto)
