nv_test(test_tensorrt_op_converter SRCS test_op_converter.cc mul_op.cc conv2d_op.cc DEPS ${FLUID_CORE_MODULES})
nv_test(test_tensorrt_activation_op SRCS test_activation_op.cc ${ENGINE_FILE} activation_op.cc 
  DEPS ${FLUID_CORE_MODULES} activation_op)
