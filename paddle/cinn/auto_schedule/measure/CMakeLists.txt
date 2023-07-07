core_gather_headers()

gather_srcs(cinnapi_src SRCS schedule_measurer.cc simple_builder.cc
            simple_runner.cc)

cinn_cc_test(test_simple_runner SRCS simple_runner_test.cc DEPS cinncore)
cinn_cc_test(test_measurer SRCS measurer_test.cc DEPS cinncore)
