cc_library(anakin_op_converter SRCS fc.cc DEPS op_registry
operator scope  framework_proto)

cc_test(test_anakin_fc SRCS test_fc_op.cc DEPS anakin_op_converter
${FLUID_CORE_MODULES} ${GLOB_OPERATOR_DEPS} mul_op anakin_engine)
