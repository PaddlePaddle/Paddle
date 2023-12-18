core_gather_headers()

gather_srcs(cinnapi_src SRCS arrange_storage_tactic.cc)

cinn_cc_test(test_arrange_storage_tactic SRCS arrange_storage_tactic_test.cc
             DEPS cinncore)
