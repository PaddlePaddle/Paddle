op_library(tensorrt_engine_op DEPS tensorrt_engine tensorrt_converter
           infer_io_utils analysis_helper)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(test_tensorrt_engine_op)
endif()
