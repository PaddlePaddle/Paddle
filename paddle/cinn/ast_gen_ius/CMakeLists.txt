core_gather_headers()

gather_srcs(cinnapi_src SRCS ast_gen.cc)

cinn_cc_test(test_ast_gen_ius SRCS ast_gen_test.cc DEPS cinncore)
