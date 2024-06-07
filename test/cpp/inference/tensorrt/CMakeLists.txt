nv_test(
  test_tensorrt_engine_instruction
  SRCS test_tensorrt_engine_instruction.cc
  DEPS pir dynload_cuda tensorrt_engine naive_executor phi)
