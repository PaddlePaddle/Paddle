paddle_test(infer_io_utils_tester SRCS io_utils_tester.cc DEPS fleet_executor
            parallel_executor)
if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(infer_io_utils_tester)
endif()
