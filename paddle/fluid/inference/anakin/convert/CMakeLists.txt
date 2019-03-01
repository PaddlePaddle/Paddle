cc_library(anakin_op_converter SRCS fc.cc conv2d.cc activation.cc DEPS anakin_engine framework_proto scope operator op_registry)
cc_test(test_anakin_fc SRCS test_fc_op.cc DEPS anakin_op_converter mul_op)
cc_test(test_anakin_conv2d SRCS test_conv2d_op.cc DEPS ${FLUID_CORE_MODULES} ${GLOB_OPERATOR_DEPS} anakin_op_converter conv_op im2col vol2col depthwise_conv SERIAL)
cc_test(test_anakin_activation SRCS test_activation_op.cc DEPS ${FLUID_CORE_MODULES} ${GLOB_OPERATOR_DEPS} activation_op anakin_op_converter SERIAL)
