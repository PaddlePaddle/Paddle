cc_library(anakin_op_converter SRCS fc.cc registrar.cc DEPS anakin_engine framework_proto scope)
cc_test(test_anakin_fc SRCS test_fc_op.cc DEPS anakin_op_converter mul_op)
