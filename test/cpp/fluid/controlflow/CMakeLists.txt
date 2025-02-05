paddle_test(conditional_block_op_test SRCS conditional_block_op_test.cc)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(conditional_block_op_test)
endif()
