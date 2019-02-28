cc_library(anakin_op_converter SRCS fc.cc conv2d.cc activation.cc registrar.cc DEPS anakin_engine framework_proto scope operator op_registry)
cc_test(test_anakin_fc SRCS test_fc_op.cc DEPS anakin_op_converter mul_op)
cc_test(test_anakin_conv2d  SRCS test_conv2d_op.cc DEPS anakin_op_converter)
cc_test(test_anakin_activation SRCS test_activation_op.cc DEPS ${GLOB_OPERATOR_DEPS} anakin_op_converter)
