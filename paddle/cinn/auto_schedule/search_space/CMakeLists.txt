add_subdirectory(auto_gen_rule)

core_gather_headers()

gather_srcs(cinnapi_src SRCS search_space.cc search_state.cc block_sampler.cc
            rule_sampler.cc)

cinn_cc_test(test_search_space SRCS search_space_test.cc DEPS cinncore)
cinn_cc_test(test_search_state SRCS search_state_test.cc DEPS cinncore)
cinn_cc_test(test_block_sampler SRCS block_sampler_test.cc DEPS cinncore)
cinn_cc_test(test_rule_sampler SRCS rule_sampler_test.cc DEPS cinncore)
