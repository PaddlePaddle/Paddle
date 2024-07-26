nv_test(
  test_tensorrt_engine_instruction
  SRCS test_tensorrt_engine_instruction.cc
  DEPS pir trt_engine naive_executor phi common)
set_tests_properties(test_tensorrt_engine_instruction PROPERTIES TIMEOUT 120)
if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(test_tensorrt_engine_instruction)
endif()
