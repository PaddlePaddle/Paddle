cc_test(dense_tensor_test SRCS dense_tensor_test.cc DEPS dense_tensor)
cc_test(kernel_factory_test SRCS kernel_factory_test.cc DEPS kernel_factory)
cc_test(test_mean_api SRCS test_mean_api.cc DEPS math_api)
cc_test(test_dot_api SRCS test_dot_api.cc DEPS dot_api)
