include(operators)
register_operators()

cc_test(test_elementwise_add_op_inplace SRCS test_elementwise_add_op_inplace.cc DEPS op_registry elementwise_add_op scope device_context enforce executor)
