nv_library(trt_convert SRCS activation_op.cc mul_op.cc io_converter.cc conv2d_op.cc)
nv_test(test_op_converter SRCS test_op_converter.cc DEPS ${FLUID_CORE_MODULES} trt_convert)
nv_test(test_io_converter SRCS test_io_converter.cc DEPS dynload_cuda dynamic_loader lod_tensor trt_convert)

# Add TRT tests
nv_test(test_trt_activation_op SRCS test_activation_op.cc
        DEPS ${FLUID_CORE_MODULES} tensorrt_engine trt_convert)
nv_test(test_trt_mul_op SRCS test_mul_op.cc
        DEPS ${FLUID_CORE_MODULES} tensorrt_engine trt_convert)
nv_test(test_op_converter SRCS test_op_converter.cc mul_op.cc conv2d_op.cc DEPS ${FLUID_CORE_MODULES})
nv_test(test_trt_activation_op SRCS test_activation_op.cc activation_op.cc io_converter.cc
  DEPS ${FLUID_CORE_MODULES} activation_op tensorrt_engine)
nv_test(test_io_converter SRCS test_io_converter.cc io_converter.cc DEPS dynload_cuda dynamic_loader lod_tensor)
>>>>>>> 6c32052653fb62b9d694a479fc83e30e1161c916
