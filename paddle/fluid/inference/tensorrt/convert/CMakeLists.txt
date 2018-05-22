nv_library(trt_convert SRCS activation_op.cc mul_op.cc io_converter.cc conv2d_op.cc)
nv_test(test_op_converter SRCS test_op_converter.cc DEPS ${FLUID_CORE_MODULES} trt_convert)
nv_test(test_io_converter SRCS test_io_converter.cc DEPS dynload_cuda dynamic_loader lod_tensor trt_convert)

# Add TRT tests
nv_test(test_trt_mul_op SRCS test_mul_op.cc mul_op.cc
        DEPS ${FLUID_CORE_MODULES} tensorrt_engine mul_op SERIAL)

nv_test(test_trt_activation_op SRCS test_activation_op.cc activation_op.cc
        DEPS ${FLUID_CORE_MODULES} tensorrt_engine activation_op SERIAL)
