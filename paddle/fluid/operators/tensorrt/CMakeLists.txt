op_library(tensorrt_engine_op DEPS tensorrt_engine tensorrt_converter)
file(APPEND ${pybind_file} "USE_NO_KERNEL_OP(tensorrt_engine);\n")
nv_test(test_tensorrt_engine_op SRCS tensorrt_engine_op_test.cc
  DEPS tensorrt_engine_op
  analysis)
