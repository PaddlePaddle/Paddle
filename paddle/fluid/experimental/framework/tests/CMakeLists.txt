cc_test(test_allocator SRCS test_allocator.cc)
cc_test(test_intrusive_ptr SRCS test_intrusive_ptr.cc)
cc_test(test_storage SRCS test_storage.cc DEPS exp_tensor_storage)
cc_test(test_dense_tensor SRCS test_dense_tensor.cc DEPS exp_dense_tensor)
