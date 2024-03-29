paddle_test(ir_kernel_dialect_pass_test SRCS ir_kernel_dialect_pass_test.cc)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(ir_kernel_dialect_pass_test)
endif()
