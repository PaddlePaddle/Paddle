add_subdirectory(mutate_rule)

core_gather_headers()

gather_srcs(cinnapi_src SRCS evolutionary_search.cc)

cinn_cc_test(test_evolutionary_search SRCS evolutionary_search_test.cc DEPS
             cinncore test_program_builder)
