include(operators)
register_operators()

cc_test(cpp_test_elementwise_add_op SRCS test_elementwise_add_op.cc DEPS op_registry elementwise_add_op scope device_context enforce executor) 
