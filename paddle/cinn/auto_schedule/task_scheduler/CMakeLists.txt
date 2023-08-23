core_gather_headers()

gather_srcs(cinnapi_src SRCS task_scheduler.cc round_robin.cc
            efficiency_priority.cc)

cinn_cc_test(test_task_scheduler SRCS task_scheduler_test.cc DEPS cinncore)
