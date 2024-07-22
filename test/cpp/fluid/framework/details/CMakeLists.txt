paddle_test(exception_holder_test SRCS exception_holder_test.cc)

cc_test(
  build_strategy_test
  SRCS build_strategy_test.cc
  DEPS build_strategy op_registry op_proto_maker graph string_helper)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(exception_holder_test)
endif()
