file(GLOB TENSORRT_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_op.cc")
nv_test(test_tensorrt_op_converter SRCS test_op_converter.cc ${TENSORRT_OPS} DEPS ${FLUID_CORE_MODULES})
