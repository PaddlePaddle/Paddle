add_subdirectory(garbage_collector)
add_subdirectory(interpreter)
add_subdirectory(workqueue)

set(STANDALONE_EXECUTOR_SRCS interpretercore.cc new_executor_defs.cc
                             standalone_executor.cc)

set(STANDALONE_EXECUTOR_DEPS interpreter interpretercore_garbage_collector
                             workqueue)

cc_library(
  standalone_executor
  SRCS ${STANDALONE_EXECUTOR_SRCS}
  DEPS ${STANDALONE_EXECUTOR_DEPS})

cc_library(
  staticgraph_executor_statistics
  SRCS executor_statistics.cc
  DEPS enforce glog phi_os_info)

# skip win32 since wget is not installed by default on windows machine.
if(WITH_GPU
   AND WITH_TESTING
   AND NOT WIN32)
  add_custom_target(
    download_program
    COMMAND wget -nc --no-check-certificate
            https://paddle-ci.gz.bcebos.com/new_exec/lm_main_program
    COMMAND wget -nc --no-check-certificate
            https://paddle-ci.gz.bcebos.com/new_exec/lm_startup_program
    WORKING_DIRECTORY "${CC_TESTS_DIR}")

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
  cc_test(standalone_executor_test SRCS standalone_executor_test.cc)

  # add_dependencies(standalone_executor_test download_program)
  # if(WITH_PROFILER)
  #   target_link_libraries(standalone_executor_test profiler)
  #   add_dependencies(standalone_executor_test profiler)
  # endif()
endif()
