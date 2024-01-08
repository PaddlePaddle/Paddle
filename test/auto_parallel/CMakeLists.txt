# file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
# string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

add_subdirectory(spmd_rules)
add_subdirectory(hybrid_strategy)
add_subdirectory(custom_op)

if(WITH_DISTRIBUTE AND WITH_GPU)

  # NOTE(zyl): unittests WITH multi cards and timeout
  py_test_modules(test_auto_parallel_relaunch MODULES
                  test_auto_parallel_relaunch)
  set_tests_properties(test_auto_parallel_relaunch
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)
  py_test_modules(test_mp_allreduce_matmul_grad_overlapping MODULES
                  test_mp_allreduce_matmul_grad_overlapping)
  set_tests_properties(test_mp_allreduce_matmul_grad_overlapping
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)
  py_test_modules(test_relaunch_with_planner MODULES test_relaunch_with_planner)
  set_tests_properties(test_relaunch_with_planner
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)
  py_test_modules(test_relaunch_with_gpt_planner MODULES
                  test_relaunch_with_gpt_planner)
  set_tests_properties(test_relaunch_with_gpt_planner
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 240)
  py_test_modules(test_engine_api MODULES test_engine_api)
  set_tests_properties(test_engine_api PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE"
                                                  TIMEOUT 80)
  py_test_modules(test_engine_api_dp MODULES test_engine_api_dp)
  set_tests_properties(test_engine_api_dp
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 80)
  py_test_modules(test_optimization_tuner_api MODULES
                  test_optimization_tuner_api)
  set_tests_properties(test_optimization_tuner_api
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)
  py_test_modules(test_converter MODULES test_converter)
  set_tests_properties(test_converter PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE"
                                                 TIMEOUT 50)
  py_test_modules(test_high_order_grad MODULES test_high_order_grad)
  set_tests_properties(test_high_order_grad
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)
  py_test_modules(test_iterable_dataset MODULES test_iterable_dataset)
  set_tests_properties(test_iterable_dataset
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 80)
  py_test_modules(test_pass_grad_clip MODULES test_pass_grad_clip)
  set_tests_properties(test_pass_grad_clip
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)
  py_test_modules(test_random_ctrl MODULES test_random_ctrl)
  set_tests_properties(test_random_ctrl PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE"
                                                   TIMEOUT 50)
  py_test_modules(test_pass_gradient_merge MODULES test_pass_gradient_merge)
  set_tests_properties(test_pass_gradient_merge
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)
  py_test_modules(test_pass_recompute MODULES test_pass_recompute)
  set_tests_properties(test_pass_recompute
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)
  py_test_modules(test_pass_sharding MODULES test_pass_sharding)
  set_tests_properties(test_pass_sharding
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  py_test_modules(test_pass_amp MODULES test_pass_amp)
  set_tests_properties(test_pass_amp PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE"
                                                TIMEOUT 80)
  py_test_modules(test_amp_o2_pass MODULES test_amp_o2_pass)
  set_tests_properties(test_amp_o2_pass PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE"
                                                   TIMEOUT 50)
  py_test_modules(test_engine_callbacks MODULES test_engine_callbacks)
  set_tests_properties(test_engine_callbacks
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)
  py_test_modules(test_pass_generation_pipeline MODULES
                  test_pass_generation_pipeline)
  set_tests_properties(test_pass_generation_pipeline
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)
  set_tests_properties(
    test_pass_generation_pipeline
    PROPERTIES ENVIRONMENT "FLAGS_new_executor_micro_batching=False")
  py_test_modules(test_pass_1F1B MODULES test_pass_1F1B)
  set_tests_properties(test_pass_1F1B PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE"
                                                 TIMEOUT 50)
  set_tests_properties(
    test_pass_1F1B PROPERTIES ENVIRONMENT
                              "FLAGS_new_executor_micro_batching=False")
  py_test_modules(test_auto_tuner MODULES test_auto_tuner)
  set_tests_properties(test_auto_tuner PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE"
                                                  TIMEOUT 100)
  py_test_modules(test_pipeline_scheduler_vpp MODULES
                  test_pipeline_scheduler_vpp)
  set_tests_properties(test_pipeline_scheduler_vpp
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)
  py_test_modules(test_auto_tuner_compare MODULES test_auto_tuner_compare)
  set_tests_properties(test_auto_tuner_compare
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  py_test_modules(test_pass_quantization MODULES test_pass_quantization)
  set_tests_properties(test_pass_quantization
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 60)
  py_test_modules(test_reshard_api MODULES test_reshard_api)
  set_tests_properties(test_reshard_api PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE"
                                                   TIMEOUT 150)
  py_test_modules(test_reshard_s_to_p MODULES test_reshard_s_to_p)
  set_tests_properties(test_reshard_s_to_p
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  py_test_modules(test_reshard_p_to_s MODULES test_reshard_p_to_s)
  set_tests_properties(test_reshard_p_to_s
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  py_test_modules(test_reshard_s_to_s MODULES test_reshard_s_to_s)
  set_tests_properties(test_reshard_s_to_s
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  if(NOT WITH_COVERAGE)
    py_test_modules(test_reshard_r_to_s MODULES test_reshard_r_to_s)
    set_tests_properties(test_reshard_r_to_s
                         PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 160)
    py_test_modules(test_reshard_p_to_r MODULES test_reshard_p_to_r)
    set_tests_properties(test_reshard_p_to_r
                         PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
    py_test_modules(test_pipeline_scheduler MODULES test_pipeline_scheduler)
    set_tests_properties(test_pipeline_scheduler
                         PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 400)
    py_test_modules(test_reshard_s_to_r MODULES test_reshard_s_to_r)
    set_tests_properties(test_reshard_s_to_r
                         PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 150)
  endif()
  py_test_modules(test_reshard_r_to_p MODULES test_reshard_r_to_p)
  set_tests_properties(test_reshard_r_to_p
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  py_test_modules(test_reshard_x_to_r MODULES test_reshard_x_to_r)
  set_tests_properties(test_reshard_x_to_r
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  py_test_modules(test_reshard_nd_mesh MODULES test_reshard_nd_mesh)
  set_tests_properties(test_reshard_nd_mesh
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)

  py_test_modules(test_reshard_same_status MODULES test_reshard_same_status)
  set_tests_properties(test_reshard_same_status
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)

  py_test_modules(test_semi_auto_parallel_basic MODULES
                  test_semi_auto_parallel_basic)
  set_tests_properties(test_semi_auto_parallel_basic
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 600)

  py_test_modules(test_semi_auto_parallel_for_llama_subnet MODULES
                  test_semi_auto_parallel_for_llama_subnet)
  set_tests_properties(test_semi_auto_parallel_for_llama_subnet
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 300)

  py_test_modules(test_semi_auto_parallel_softmax_basic MODULES
                  test_semi_auto_parallel_softmax_basic)
  set_tests_properties(test_semi_auto_parallel_softmax_basic
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)

  py_test_modules(test_semi_auto_parallel_compare_basic MODULES
                  test_semi_auto_parallel_compare_basic)
  set_tests_properties(test_semi_auto_parallel_compare_basic
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)
  py_test_modules(test_semi_auto_parallel_subgraph_embedding_basic MODULES
                  test_semi_auto_parallel_subgraph_embedding_basic)
  set_tests_properties(test_semi_auto_parallel_subgraph_embedding_basic
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)
  py_test_modules(test_semi_auto_parallel_pylayer MODULES
                  test_semi_auto_parallel_pylayer)
  set_tests_properties(test_semi_auto_parallel_pylayer
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  py_test_modules(test_semi_auto_parallel_single_strategy MODULES
                  test_semi_auto_parallel_single_strategy)
  set_tests_properties(test_semi_auto_parallel_single_strategy
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 400)
  py_test_modules(test_semi_auto_parallel_lazy_init MODULES
                  test_semi_auto_parallel_lazy_init)
  set_tests_properties(test_semi_auto_parallel_lazy_init
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)
  py_test_modules(test_semi_auto_parallel_in_framework MODULES
                  test_semi_auto_parallel_in_framework)
  set_tests_properties(test_semi_auto_parallel_in_framework
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 300)
  py_test_modules(test_semi_auto_parallel_dygraph_inplace MODULES
                  test_semi_auto_parallel_dygraph_inplace)
  set_tests_properties(test_semi_auto_parallel_dygraph_inplace
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  py_test_modules(test_dist_tensor_api MODULES test_dist_tensor_api)
  set_tests_properties(test_dist_tensor_api
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  py_test_modules(test_gpt_with_pir MODULES test_gpt_with_pir)
  set_tests_properties(test_gpt_with_pir PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE"
                                                    TIMEOUT 100)
  py_test_modules(test_semi_auto_parallel_saved_tensor_hook MODULES
                  test_semi_auto_parallel_saved_tensor_hook)
  set_tests_properties(test_semi_auto_parallel_saved_tensor_hook
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  py_test_modules(test_semi_auto_parallel_dist_to_static MODULES
                  test_semi_auto_parallel_dist_to_static)
  set_tests_properties(test_semi_auto_parallel_dist_to_static
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 300)
  py_test_modules(test_static_reshard_api MODULES test_static_reshard_api)
  set_tests_properties(test_static_reshard_api
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 300)
  py_test_modules(test_gpt_with_prim MODULES test_gpt_with_prim)
  set_tests_properties(test_gpt_with_prim
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 200)
  py_test_modules(test_dist_checkpoint_utils MODULES test_dist_checkpoint_utils)
  set_tests_properties(test_dist_checkpoint_utils
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)
  py_test_modules(test_semi_auto_parallel_unshard_dtensor MODULES
                  test_semi_auto_parallel_unshard_dtensor)
  set_tests_properties(test_semi_auto_parallel_unshard_dtensor
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  # End of unittests WITH multi cards and timeout

  # NOTE(zyl): unittests WITH multi cards and WITHOUT timeout
  # End of unittests WITH multi cards and WITHOUT timeout

  # NOTE(zyl): unittests WITH single card and timeout
  py_test_modules(test_parallel_tuner MODULES test_parallel_tuner)
  set_tests_properties(test_parallel_tuner PROPERTIES TIMEOUT 120)
  py_test_modules(test_parallel_tuner_full MODULES test_parallel_tuner_full)
  set_tests_properties(test_parallel_tuner_full PROPERTIES TIMEOUT 120)
  py_test_modules(test_parallel_tuner_predict MODULES
                  test_parallel_tuner_predict)
  set_tests_properties(test_parallel_tuner_predict PROPERTIES TIMEOUT 120)
  py_test_modules(test_selective_recompute MODULES test_selective_recompute)
  set_tests_properties(test_selective_recompute PROPERTIES TIMEOUT 50)
  py_test_modules(test_tuning_recompute MODULES test_tuning_recompute)
  set_tests_properties(test_tuning_recompute PROPERTIES TIMEOUT 300)
  py_test_modules(test_tuning_recompute_with_amp MODULES
                  test_tuning_recompute_with_amp)
  set_tests_properties(test_tuning_recompute_with_amp PROPERTIES TIMEOUT 60)
  py_test_modules(test_fused_linear_pass MODULES test_fused_linear_pass)
  set_tests_properties(test_fused_linear_pass PROPERTIES TIMEOUT 40)
  py_test_modules(test_align_tool MODULES test_align_tool)
  set_tests_properties(test_align_tool PROPERTIES TIMEOUT 20)
  py_test_modules(test_pass_base_list MODULES test_pass_base_list)
  set_tests_properties(test_pass_base_list PROPERTIES TIMEOUT 40)
  py_test_modules(test_fuse_adamw_pass MODULES test_fuse_adamw_pass)
  set_tests_properties(test_fuse_adamw_pass PROPERTIES TIMEOUT 20)
  py_test_modules(test_rule_based_tuner_o2 MODULES test_rule_based_tuner_o2)
  set_tests_properties(test_rule_based_tuner_o2 PROPERTIES TIMEOUT 50)
  py_test_modules(test_semi_auto_parallel_functional_in_single_card MODULES
                  test_semi_auto_parallel_functional_in_single_card)
  # End of unittests WITH single card and timeout

  # NOTE(zyl): unittests WITH single card and WITHOUT timeout
  py_test_modules(test_while_op_completion MODULES test_while_op_completion)
  py_test_modules(test_while_op_partition MODULES test_while_op_partition)
  py_test_modules(test_tunable_variable MODULES test_tunable_variable)
  py_test_modules(test_tunable_space MODULES test_tunable_space)
  py_test_modules(test_recorder MODULES test_recorder)
  py_test_modules(test_trial MODULES test_trial)
  py_test_modules(test_new_cost_model MODULES test_new_cost_model)
  py_test_modules(test_dist_reshape MODULES test_dist_reshape)
  py_test_modules(test_dist_pnorm MODULES test_dist_pnorm)
  py_test_modules(test_dist_embedding MODULES test_dist_embedding)
  py_test_modules(test_dist_slice MODULES test_dist_slice)
  py_test_modules(test_dist_split MODULES test_dist_split)
  py_test_modules(test_cluster MODULES test_cluster)
  py_test_modules(test_comm_cost MODULES test_comm_cost)
  py_test_modules(test_comp_cost MODULES test_comp_cost)
  py_test_modules(test_base_cost MODULES test_base_cost)
  py_test_modules(test_dist_context MODULES test_dist_context)
  py_test_modules(test_prim_dist_op MODULES test_prim_dist_op)
  py_test_modules(test_to_static MODULES test_to_static)
  py_test_modules(test_dist_op_cost MODULES test_dist_op_cost)
  py_test_modules(test_cluster_v2 MODULES test_cluster_v2)
  py_test_modules(test_process_mesh_v2 MODULES test_process_mesh_v2)
  py_test_modules(test_dist_attr_v2 MODULES test_dist_attr_v2)
  py_test_modules(test_serialization MODULES test_serialization)
  py_test_modules(test_lr_grad_clip MODULES test_lr_grad_clip)
  py_test_modules(test_dist_matmul MODULES test_dist_matmul)
  py_test_modules(test_process_mesh MODULES test_process_mesh)
  py_test_modules(test_interface MODULES test_interface)
  py_test_modules(test_strategy MODULES test_strategy)
  py_test_modules(test_dist_shape MODULES test_dist_shape)
  py_test_modules(test_dist_assign MODULES test_dist_assign)
  py_test_modules(test_conditional_block_reshard MODULES
                  test_conditional_block_reshard)
  py_test_modules(test_engine_api_error MODULES test_engine_api_error)
  py_test_modules(test_fp16_assign MODULES test_fp16_assign)
  py_test_modules(test_group_operators MODULES test_group_operators)
  py_test_modules(test_pattern MODULES test_pattern)
  py_test_modules(test_pattern_match MODULES test_pattern_match)
  py_test_modules(test_cluster_partition MODULES test_cluster_partition)
  py_test_modules(test_convert_to_process_meshes MODULES
                  test_convert_to_process_meshes)
  py_test_modules(test_pass_bf16 MODULES test_pass_bf16)
  py_test_modules(test_dist_saver MODULES test_dist_saver)
  py_test_modules(test_engine_save_load MODULES test_engine_save_load)
  py_test_modules(test_rule_based_tuner MODULES test_rule_based_tuner)
  py_test_modules(test_dist_tensor MODULES test_dist_tensor)
  py_test_modules(test_api_dist_branch MODULES test_api_dist_branch)
  py_test_modules(test_shard_tensor_api MODULES test_shard_tensor_api)
  py_test_modules(test_shard_layer_api MODULES test_shard_layer_api)
  py_test_modules(test_cost_interface MODULES test_cost_interface)
  py_test_modules(test_auto_conditional_block MODULES
                  test_auto_conditional_block)
  py_test_modules(test_strategy_api MODULES test_strategy_api)
  py_test_modules(test_dist_concat MODULES test_dist_concat)
  # End of unittests WITH single card WITHOUT timeout

endif()

py_test_modules(test_job_schedule_profiler_range MODULES
                test_job_schedule_profiler_range)
