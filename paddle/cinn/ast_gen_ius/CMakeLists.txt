core_gather_headers()

gather_srcs(cinnapi_src SRCS ast_gen.cc tensor_group.cc)

cinn_cc_test(test_ast_gen_ius SRCS ast_gen_test.cc DEPS cinncore)
cinn_cc_test(test_tensor_group SRCS tensor_group_test.cc DEPS cinncore)
