if(WITH_TESTING)
  if(NOT APPLE)
    inference_base_test(infer_io_utils_tester SRCS io_utils_tester.cc DEPS
                        paddle_inference_shared common)
  endif()
endif()

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(infer_io_utils_tester)
endif()
paddle_test(test_benchmark SRCS benchmark_tester.cc DEPS benchmark)
paddle_test(test_table_printer SRCS table_printer_tester.cc DEPS pir)
