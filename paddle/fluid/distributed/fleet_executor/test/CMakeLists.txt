set_source_files_properties(
  interceptor_ping_pong_test.cc PROPERTIES COMPILE_FLAGS
                                           ${DISTRIBUTE_COMPILE_FLAGS})
<<<<<<< HEAD
cc_test_old(interceptor_ping_pong_test SRCS interceptor_ping_pong_test.cc DEPS
            fleet_executor ${BRPC_DEPS})
=======
cc_test(
  interceptor_ping_pong_test
  SRCS interceptor_ping_pong_test.cc
  DEPS fleet_executor ${BRPC_DEPS})
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e

set_source_files_properties(
  compute_interceptor_test.cc PROPERTIES COMPILE_FLAGS
                                         ${DISTRIBUTE_COMPILE_FLAGS})
<<<<<<< HEAD
cc_test_old(compute_interceptor_test SRCS compute_interceptor_test.cc DEPS
            fleet_executor ${BRPC_DEPS})
=======
cc_test(
  compute_interceptor_test
  SRCS compute_interceptor_test.cc
  DEPS fleet_executor ${BRPC_DEPS})
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e

set_source_files_properties(
  source_interceptor_test.cc PROPERTIES COMPILE_FLAGS
                                        ${DISTRIBUTE_COMPILE_FLAGS})
<<<<<<< HEAD
cc_test_old(source_interceptor_test SRCS source_interceptor_test.cc DEPS
            fleet_executor ${BRPC_DEPS})

set_source_files_properties(
  sink_interceptor_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test_old(sink_interceptor_test SRCS sink_interceptor_test.cc DEPS
            fleet_executor ${BRPC_DEPS})
=======
cc_test(
  source_interceptor_test
  SRCS source_interceptor_test.cc
  DEPS fleet_executor ${BRPC_DEPS})

set_source_files_properties(
  sink_interceptor_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(
  sink_interceptor_test
  SRCS sink_interceptor_test.cc
  DEPS fleet_executor ${BRPC_DEPS})
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e

set_source_files_properties(
  interceptor_pipeline_short_path_test.cc
  PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
<<<<<<< HEAD
cc_test_old(
  interceptor_pipeline_short_path_test SRCS
  interceptor_pipeline_short_path_test.cc DEPS fleet_executor ${BRPC_DEPS})
=======
cc_test(
  interceptor_pipeline_short_path_test
  SRCS interceptor_pipeline_short_path_test.cc
  DEPS fleet_executor ${BRPC_DEPS})
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e

set_source_files_properties(
  interceptor_pipeline_long_path_test.cc PROPERTIES COMPILE_FLAGS
                                                    ${DISTRIBUTE_COMPILE_FLAGS})
<<<<<<< HEAD
cc_test_old(
  interceptor_pipeline_long_path_test SRCS
  interceptor_pipeline_long_path_test.cc DEPS fleet_executor ${BRPC_DEPS})
=======
cc_test(
  interceptor_pipeline_long_path_test
  SRCS interceptor_pipeline_long_path_test.cc
  DEPS fleet_executor ${BRPC_DEPS})
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e

set_source_files_properties(
  compute_interceptor_run_op_test.cc PROPERTIES COMPILE_FLAGS
                                                ${DISTRIBUTE_COMPILE_FLAGS})
<<<<<<< HEAD
cc_test_old(
  compute_interceptor_run_op_test
  SRCS
  compute_interceptor_run_op_test.cc
  DEPS
  fleet_executor
  ${BRPC_DEPS}
  op_registry
  fill_constant_op
  elementwise_add_op
  scope
  device_context)
=======
cc_test(
  compute_interceptor_run_op_test
  SRCS compute_interceptor_run_op_test.cc
  DEPS fleet_executor
       ${BRPC_DEPS}
       op_registry
       fill_constant_op
       elementwise_add_op
       scope
       device_context)
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e

if(WITH_DISTRIBUTE
   AND WITH_PSCORE
   AND NOT (WITH_ASCEND OR WITH_ASCEND_CL))
  set_source_files_properties(
    interceptor_ping_pong_with_brpc_test.cc
    PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
<<<<<<< HEAD
  cc_test_old(
    interceptor_ping_pong_with_brpc_test SRCS
    interceptor_ping_pong_with_brpc_test.cc DEPS fleet_executor ${BRPC_DEPS})
=======
  cc_test(
    interceptor_ping_pong_with_brpc_test
    SRCS interceptor_ping_pong_with_brpc_test.cc
    DEPS fleet_executor ${BRPC_DEPS})
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
endif()
