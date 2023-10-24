core_gather_headers()

gather_srcs(cinnapi_src SRCS analyze_ir.cc)

cinn_cc_test(test_analyze_ir SRCS analyze_ir_test.cc DEPS cinncore)
