paddle_test(broadcast_op_test SRCS broadcast_op_handle_test.cc)

paddle_test(fused_broadcast_op_test SRCS fused_broadcast_op_handle_test.cc)
paddle_test(exception_holder_test SRCS exception_holder_test.cc)

cc_test(
  build_strategy_test
  SRCS build_strategy_test.cc
  DEPS build_strategy op_registry op_proto_maker graph string_helper)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(broadcast_op_test)
  copy_onnx(fused_broadcast_op_test)
  copy_onnx(exception_holder_test)
endif()
