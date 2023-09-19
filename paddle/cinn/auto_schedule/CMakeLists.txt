add_subdirectory(analysis)
add_subdirectory(cost_model)
add_subdirectory(database)
add_subdirectory(measure)
add_subdirectory(post_schedule_rule)
add_subdirectory(search_space)
add_subdirectory(search_strategy)
add_subdirectory(task)
add_subdirectory(task_scheduler)
add_subdirectory(tests)

cinn_proto_library(auto_schedule_proto SRCS auto_schedule.proto DEPS
                   schedule_desc_proto)

core_gather_headers()

gather_srcs(cinnapi_src SRCS auto_tuner.cc)

#cinn_cc_test(test_auto_tuner SRCS auto_tuner_test.cc DEPS cinncore)

foreach(header ${auto_schedule_proto_HDRS})
  set(core_proto_includes
      "${core_proto_includes};${header}"
      CACHE INTERNAL "")
endforeach()
