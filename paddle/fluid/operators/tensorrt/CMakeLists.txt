set(tensorrt_engine_op_deps tensorrt_engine tensorrt_converter infer_io_utils
                            analysis_helper)

op_library(tensorrt_engine_op DEPS ${tensorrt_engine_op_deps})
nv_test(
  test_tensorrt_engine_op
  SRCS tensorrt_engine_op_test.cc
  DEPS tensorrt_engine_op analysis fleet_executor python)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(test_tensorrt_engine_op)
endif()
