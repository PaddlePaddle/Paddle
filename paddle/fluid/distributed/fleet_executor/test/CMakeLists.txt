# NOTE(risemeup1):The compile target of these unit tests requires occupying more than 4GB of memory,
# which lead a intractable compilation failure issue on Coverage CI.
# We temporarily disabled them, and will restore after all compilation issues are resolved.

# get_property(paddle_lib GLOBAL PROPERTY PADDLE_LIB_NAME)
# set_source_files_properties(
#   interceptor_ping_pong_test.cc PROPERTIES COMPILE_FLAGS
#                                            ${DISTRIBUTE_COMPILE_FLAGS})
# if(WIN32 AND WITH_TESTING)
#   cc_test_old(interceptor_ping_pong_test SRCS interceptor_ping_pong_test.cc
#               DEPS fleet_executor ${BRPC_DEPS})
# else()
#   cc_test_old(
#     interceptor_ping_pong_test
#     SRCS
#     interceptor_ping_pong_test.cc
#     DEPS
#     ${paddle_lib}
#     python
#     fleet_executor)
# endif()

# set_source_files_properties(
#   compute_interceptor_test.cc PROPERTIES COMPILE_FLAGS
#                                          ${DISTRIBUTE_COMPILE_FLAGS})

# if(WIN32 AND WITH_TESTING)
#   cc_test_old(compute_interceptor_test SRCS compute_interceptor_test.cc DEPS
#               fleet_executor ${BRPC_DEPS})
# else()
#   cc_test_old(compute_interceptor_test SRCS compute_interceptor_test.cc DEPS
#               ${paddle_lib} python)
# endif()

# set_source_files_properties(
#   source_interceptor_test.cc PROPERTIES COMPILE_FLAGS
#                                         ${DISTRIBUTE_COMPILE_FLAGS})
# if(WIN32 AND WITH_TESTING)
#   cc_test_old(source_interceptor_test SRCS source_interceptor_test.cc DEPS
#               fleet_executor ${BRPC_DEPS})
# else()
#   cc_test_old(source_interceptor_test SRCS source_interceptor_test.cc DEPS
#               ${paddle_lib} python)
# endif()

# set_source_files_properties(
#   sink_interceptor_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
# if(WIN32 AND WITH_TESTING)
#   cc_test_old(sink_interceptor_test SRCS sink_interceptor_test.cc DEPS
#               fleet_executor ${BRPC_DEPS})
# else()
#   cc_test_old(sink_interceptor_test SRCS sink_interceptor_test.cc DEPS
#               ${paddle_lib} python)
# endif()

# set_source_files_properties(
#   interceptor_pipeline_short_path_test.cc
#   PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
# if(WIN32 AND WITH_TESTING)
#   cc_test_old(
#     interceptor_pipeline_short_path_test SRCS
#     interceptor_pipeline_short_path_test.cc DEPS fleet_executor ${BRPC_DEPS})
# else()
#   cc_test_old(interceptor_pipeline_short_path_test SRCS
#               interceptor_pipeline_short_path_test.cc DEPS ${paddle_lib} python)
# endif()

# set_source_files_properties(
#   interceptor_pipeline_long_path_test.cc PROPERTIES COMPILE_FLAGS
#                                                     ${DISTRIBUTE_COMPILE_FLAGS})
# if(WIN32 AND WITH_TESTING)
#   cc_test_old(
#     interceptor_pipeline_long_path_test SRCS
#     interceptor_pipeline_long_path_test.cc DEPS fleet_executor ${BRPC_DEPS})
# else()
#   cc_test_old(interceptor_pipeline_long_path_test SRCS
#               interceptor_pipeline_long_path_test.cc DEPS ${paddle_lib} python)
# endif()

# set_source_files_properties(
#   compute_interceptor_run_op_test.cc PROPERTIES COMPILE_FLAGS
#                                                 ${DISTRIBUTE_COMPILE_FLAGS})
# if(WIN32 AND WITH_TESTING)
#   cc_test_old(
#     compute_interceptor_run_op_test
#     SRCS
#     compute_interceptor_run_op_test.cc
#     DEPS
#     fleet_executor
#     naive_executor
#     fill_constant_op
#     op_registry
#     elementwise_add_op
#     scope
#     device_context
#     ${BRPC_DEPS})
# else()
#   cc_test_old(
#     compute_interceptor_run_op_test
#     SRCS
#     compute_interceptor_run_op_test.cc
#     DEPS
#     ${paddle_lib}
#     python
#     fill_constant_op
#     elementwise_add_op)
# endif()

# if(WITH_DISTRIBUTE AND NOT WITH_PSLIB)
#   set_source_files_properties(
#     interceptor_ping_pong_with_brpc_test.cc
#     PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
#   if(WIN32 AND WITH_TESTING)
#     cc_test_old(
#       interceptor_ping_pong_with_brpc_test SRCS
#       interceptor_ping_pong_with_brpc_test.cc DEPS fleet_executor ${BRPC_DEPS})
#   else()
#     cc_test_old(
#       interceptor_ping_pong_with_brpc_test SRCS
#       interceptor_ping_pong_with_brpc_test.cc DEPS ${paddle_lib} python)
#   endif()
# endif()
