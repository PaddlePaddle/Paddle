set_source_files_properties(interceptor_ping_pong_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(interceptor_ping_pong_test SRCS interceptor_ping_pong_test.cc DEPS fleet_executor ${BRPC_DEPS})
