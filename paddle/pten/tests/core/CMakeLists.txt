cc_test(test_allocator SRCS test_allocator.cc DEPS tensor_base)
cc_test(test_storage SRCS test_storage.cc DEPS tensor_base)
cc_test(test_dense_tensor SRCS test_dense_tensor.cc DEPS dense_tensor)
