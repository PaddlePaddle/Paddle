# Add TRT tests
nv_library(tensorrt_converter
           SRCS mul_op.cc conv2d_op.cc fc_op.cc pool2d_op.cc elementwise_op.cc
                batch_norm_op.cc activation_op.cc softmax_op.cc concat_op.cc dropout_op.cc
                pad_op.cc split_op.cc prelu_op.cc leaky_relu_op.cc
           DEPS tensorrt_engine tensorrt_plugin operator scope framework_proto op_registry)

nv_test(test_op_converter SRCS test_op_converter.cc DEPS
  ${FLUID_CORE_MODULES} ${GLOB_OPERATOR_DEPS} tensorrt_engine tensorrt_converter)

nv_test(test_io_converter SRCS test_io_converter.cc io_converter.cc DEPS dynload_cuda dynamic_loader lod_tensor)
nv_test(test_trt_mul_op SRCS test_mul_op.cc mul_op.cc
        DEPS ${FLUID_CORE_MODULES} ${GLOB_OPERATOR_DEPS} tensorrt_engine mul_op)
nv_test(test_trt_fc_op SRCS test_fc_op.cc fc_op.cc
        DEPS ${FLUID_CORE_MODULES} ${GLOB_OPERATOR_DEPS} tensorrt_engine mul_op)
nv_test(test_trt_activation_op SRCS test_activation_op.cc activation_op.cc
        DEPS ${FLUID_CORE_MODULES} ${GLOB_OPERATOR_DEPS} tensorrt_engine activation_op)
nv_test(test_trt_conv_op SRCS test_conv2d_op.cc conv2d_op.cc
        DEPS ${FLUID_CORE_MODULES} ${GLOB_OPERATOR_DEPS} tensorrt_engine conv_op conv_transpose_op)
nv_test(test_trt_pool2d_op SRCS test_pool2d_op.cc pool2d_op.cc
        DEPS ${FLUID_CORE_MODULES} ${GLOB_OPERATOR_DEPS} tensorrt_engine pool_op tensorrt_plugin)
nv_test(test_trt_elementwise_op SRCS test_elementwise_op.cc elementwise_op.cc
        DEPS ${FLUID_CORE_MODULES} ${GLOB_OPERATOR_DEPS} tensorrt_engine tensorrt_plugin
             elementwise_add_op elementwise_mul_op)
nv_test(test_trt_softmax_op SRCS test_softmax_op.cc softmax_op.cc
        DEPS ${FLUID_CORE_MODULES} ${GLOB_OPERATOR_DEPS} tensorrt_engine softmax_op)
nv_test(test_trt_batch_norm_op SRCS test_batch_norm_op.cc batch_norm_op.cc
        DEPS ${FLUID_CORE_MODULES} ${GLOB_OPERATOR_DEPS} tensorrt_engine batch_norm_op)
nv_test(test_trt_concat_op SRCS test_concat_op.cc concat_op.cc
        DEPS ${FLUID_CORE_MODULES} ${GLOB_OPERATOR_DEPS} tensorrt_engine concat_op)
nv_test(test_trt_dropout_op SRCS test_dropout_op.cc dropout_op.cc
        DEPS ${FLUID_CORE_MODULES} ${GLOB_OPERATOR_DEPS} tensorrt_engine dropout_op)
nv_test(test_trt_pad_op SRCS test_pad_op.cc pad_op.cc
        DEPS ${FLUID_CORE_MODULES} ${GLOB_OPERATOR_DEPS} tensorrt_engine pad_op)
nv_test(test_trt_split_op SRCS test_split_op.cc split_op.cc
        DEPS ${FLUID_CORE_MODULES} ${GLOB_OPERATOR_DEPS} tensorrt_engine tensorrt_plugin
             split_op concat_op)
nv_test(test_trt_prelu_op SRCS test_prelu_op.cc prelu_op.cc
        DEPS ${FLUID_CORE_MODULES} ${GLOB_OPERATOR_DEPS} tensorrt_engine tensorrt_plugin
        prelu_op)
nv_test(test_trt_leaky_relu_op SRCS test_leaky_relu_op.cc leaky_relu_op.cc
        DEPS ${FLUID_CORE_MODULES} ${GLOB_OPERATOR_DEPS} tensorrt_engine activation_op)
