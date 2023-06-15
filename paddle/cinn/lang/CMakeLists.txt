core_gather_headers()

gather_srcs(cinnapi_src SRCS
    buffer.cc
    compute.cc
    placeholder.cc
    lower.cc
    builtin.cc
    lower_impl.cc
    packed_func.cc
    )

cc_test(test_compute SRCS compute_test.cc DEPS cinncore)
cc_test(test_placeholder SRCS placeholder_test.cc DEPS cinncore)
cc_test(test_lower SRCS lower_test.cc DEPS cinncore)
cc_test(test_lower_impl SRCS lower_impl_test.cc DEPS cinncore)
cc_test(test_cinn_packed_func SRCS packed_func_test.cc DEPS cinncore)
