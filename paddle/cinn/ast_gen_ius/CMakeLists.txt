core_gather_headers()

gather_srcs(cinnapi_src SRCS ast_gen.cc tensor_group.cc)

cinn_cc_test(test_tensor_group SRCS tensor_group_test.cc DEPS cinncore)
