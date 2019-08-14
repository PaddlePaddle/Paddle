cc_test(nccl_context_test SRCS nccl_context_test.cc DEPS nccl_context)
cc_test(test_add_functor SRCS test_add_functor.cc DEPS gradient_accumulator memcpy)
