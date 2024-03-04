paddle_test(test_lite_engine_op SRCS lite_engine_op_test.cc)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(test_lite_engine_op)
endif()
