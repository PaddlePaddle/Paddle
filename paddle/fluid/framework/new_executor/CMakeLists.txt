add_subdirectory(garbage_collector)
add_subdirectory(interpreter)
add_subdirectory(workqueue)

set(STANDALONE_EXECUTOR_SRCS
    data_transfer.cc
    new_executor_defs.cc
    interpretercore_util.cc
    event_manager.cc
    stream_analyzer.cc
    interpretercore.cc
    standalone_executor.cc)

set(STANDALONE_EXECUTOR_DEPS
    dependency_builder
    device_context
    op_registry
    scope
    framework_proto
    data_feed_proto
    ops_extra_info
    heter_service_proto
    trainer_desc_proto
    glog
    lod_rank_table
    fs
    shell
    fleet_wrapper
    heter_wrapper
    ps_gpu_wrapper
    box_wrapper
    lodtensor_printer
    feed_fetch_method
    graph_to_program_pass
    variable_helper
    timer
    monitor
    nan_inf_utils
    enforce
    scope
    glog
    workqueue
    interpretercore_garbage_collector
    ${DEVICE_EVENT_LIBS}
    glog)

cc_library(
  standalone_executor
  SRCS ${STANDALONE_EXECUTOR_SRCS}
  DEPS ${STANDALONE_EXECUTOR_DEPS})

cc_library(
  staticgraph_executor_statistics
  SRCS executor_statistics.cc
  DEPS enforce glog os_info)

# skip win32 since wget is not installed by default on windows machine.
if(WITH_GPU
   AND WITH_TESTING
   AND NOT WIN32)
  add_custom_target(
    download_program
    COMMAND wget -nc --no-check-certificate
            https://paddle-ci.gz.bcebos.com/new_exec/lm_main_program
    COMMAND wget -nc --no-check-certificate
            https://paddle-ci.gz.bcebos.com/new_exec/lm_startup_program)

  # all operators used in the program
  set(OPS
      fill_constant_op
      uniform_random_op
      lookup_table_op
      transpose_op
      reshape_op
      split_op
      slice_op
      concat_op
      matmul_op
      elementwise_add_op
      elementwise_mul_op
      softmax_with_cross_entropy_op
      reduce_mean_op
      reduce_sum_op
      activation_op
      sum_op
      elementwise_max_op
      elementwise_div_op
      sgd_op
      squared_l2_norm_op
      memcpy_h2d_op
      memcpy_d2h_op
      fetch_v2_op)

  # All deps of the operators above, part of GLOB_OPERATOR_DEPS.
  set(OP_DEPS generator softmax selected_rows_functor jit_kernel_helper
              concat_and_split cross_entropy)

  cc_test(
    standalone_executor_test
    SRCS standalone_executor_test.cc
    DEPS standalone_executor operator op_registry executor ${OPS} ${OP_DEPS})
  set_tests_properties(standalone_executor_test PROPERTIES TIMEOUT 100)

  add_dependencies(standalone_executor_test download_program)
  if(WITH_PROFILER)
    target_link_libraries(standalone_executor_test profiler)
    add_dependencies(standalone_executor_test profiler)
  endif()
endif()
