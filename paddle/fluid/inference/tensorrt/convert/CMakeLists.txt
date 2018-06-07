# Add TRT tests
# This test is not stable
# See https://paddleci.ngrok.io/viewLog.html?tab=buildLog&buildTypeId=Paddle_PrCi2&buildId=36834&_focus=8828 
#nv_test(test_trt_activation_op SRCS test_activation_op.cc activation_op.cc io_converter.cc
#    DEPS ${FLUID_CORE_MODULES} activation_op tensorrt_engine
#    SERIAL)
nv_library(tensorrt_converter
  SRCS mul_op.cc conv2d_op.cc fc_op.cc
  DEPS tensorrt_engine mul_op)

nv_test(test_op_converter SRCS test_op_converter.cc DEPS
  ${FLUID_CORE_MODULES} tensorrt_engine tensorrt_converter)

nv_test(test_io_converter SRCS test_io_converter.cc io_converter.cc DEPS dynload_cuda dynamic_loader lod_tensor)
nv_test(test_trt_mul_op SRCS test_mul_op.cc mul_op.cc
        DEPS ${FLUID_CORE_MODULES} tensorrt_engine mul_op SERIAL)
nv_test(test_trt_fc_op SRCS test_fc_op.cc fc_op.cc
        DEPS ${FLUID_CORE_MODULES} tensorrt_engine mul_op SERIAL)
