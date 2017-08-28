py_test(test_net SRCS test_net.py)

py_test(test_scope SRCS test_scope.py)

py_test(test_tensor SRCS test_tensor.py)
py_test(test_mul_op SRCS test_mul_op.py)

py_test(test_mean_op SRCS test_mean_op.py)

py_test(test_protobuf SRCS test_protobuf.py)

py_test(test_add_two_op SRCS test_add_two_op.py)
py_test(test_sigmoid_op SRCS test_sigmoid_op.py)
py_test(test_softmax_op SRCS test_softmax_op.py)
py_test(test_cross_entropy_op SRCS test_cross_entropy_op.py)
py_test(test_gather_op SRCS test_gather_op.py)
py_test(test_scatter_op SRCS test_scatter_op.py)
py_test(test_fill_zeros_like_op SRCS test_fill_zeros_like_op.py)

py_test(gradient_checker SRCS gradient_checker.py)

py_test(test_rowwise_add_op SRCS test_rowwise_add_op.py)

py_test(test_default_scope_funcs SRCS test_default_scope_funcs.py)

py_test(test_operator SRCS test_operator.py)
py_test(test_gaussian_random_op SRCS test_gaussian_random_op.py)
py_test(test_uniform_random_op SRCS test_uniform_random_op.py)
py_test(test_recurrent_op SRCS test_recurrent_op.py)
py_test(test_sgd_op SRCS test_sgd_op.py)
py_test(test_gradient_checker SRCS test_gradient_checker.py)
py_test(test_lookup_table SRCS test_lookup_table.py)
py_test(test_scale_and_identity_op SRCS test_scale_and_identity_op.py)
py_test(mnist SRCS mnist.py)
