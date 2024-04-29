core_gather_headers()

gather_srcs(cinnapi_src SRCS mutate_rule.cc mutate_tile_size.cc)

cinn_cc_test(test_mutate_tile_size SRCS mutate_tile_size_test.cc DEPS cinncore)
