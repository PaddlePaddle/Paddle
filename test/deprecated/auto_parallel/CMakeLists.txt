# file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
# string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

# add_subdirectory(spmd_rules)
# add_subdirectory(hybrid_strategy)
# add_subdirectory(custom_op)
# add_subdirectory(pir)

if(WITH_DISTRIBUTE AND WITH_GPU)
  py_test_modules(test_auto_parallel_relaunch_deprecated MODULES
                  test_auto_parallel_relaunch_deprecated)
  set_tests_properties(test_auto_parallel_relaunch_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)
  py_test_modules(test_engine_api_dp_deprecated MODULES
                  test_engine_api_dp_deprecated)
  set_tests_properties(test_engine_api_dp_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 80)
  py_test_modules(test_engine_api_deprecated MODULES test_engine_api_deprecated)
  set_tests_properties(test_engine_api_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 80)
  py_test_modules(test_auto_tuner_compare_deprecated MODULES
                  test_auto_tuner_compare_deprecated)
  set_tests_properties(test_auto_tuner_compare_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  py_test_modules(test_auto_tuner_deprecated MODULES test_auto_tuner_deprecated)
  set_tests_properties(test_auto_tuner_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  py_test_modules(test_optimization_tuner_api_deprecated MODULES
                  test_optimization_tuner_api_deprecated)
  set_tests_properties(test_optimization_tuner_api_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)
  py_test_modules(test_relaunch_with_planner_deprecated MODULES
                  test_relaunch_with_planner_deprecated)
  set_tests_properties(test_relaunch_with_planner_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)
  py_test_modules(test_random_ctrl_deprecated MODULES
                  test_random_ctrl_deprecated)
  set_tests_properties(test_random_ctrl_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)
  py_test_modules(test_pipeline_scheduler_vpp_deprecated MODULES
                  test_pipeline_scheduler_vpp_deprecated)
  set_tests_properties(test_pipeline_scheduler_vpp_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)
  py_test_modules(test_pipeline_scheduler_zb_deprecated MODULES
                  test_pipeline_scheduler_zb_deprecated)
  set_tests_properties(test_pipeline_scheduler_zb_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)
  py_test_modules(test_selective_recompute_deprecated MODULES
                  test_selective_recompute_deprecated)
  set_tests_properties(test_selective_recompute_deprecated PROPERTIES TIMEOUT
                                                                      50)
  py_test_modules(test_parallel_tuner_deprecated MODULES
                  test_parallel_tuner_deprecated)
  set_tests_properties(test_parallel_tuner_deprecated PROPERTIES TIMEOUT 120)
  py_test_modules(test_parallel_tuner_full_deprecated MODULES
                  test_parallel_tuner_full_deprecated)
  set_tests_properties(test_parallel_tuner_full_deprecated PROPERTIES TIMEOUT
                                                                      120)
  py_test_modules(test_parallel_tuner_predict_deprecated MODULES
                  test_parallel_tuner_predict_deprecated)
  set_tests_properties(test_parallel_tuner_predict_deprecated PROPERTIES TIMEOUT
                                                                         120)
  py_test_modules(test_fused_linear_pass_deprecated MODULES
                  test_fused_linear_pass_deprecated)
  set_tests_properties(test_fused_linear_pass_deprecated PROPERTIES TIMEOUT 40)
  py_test_modules(test_fuse_adamw_pass_deprecated MODULES
                  test_fuse_adamw_pass_deprecated)
  set_tests_properties(test_fuse_adamw_pass_deprecated PROPERTIES TIMEOUT 20)
  py_test_modules(test_engine_callbacks_deprecated MODULES
                  test_engine_callbacks_deprecated)
  set_tests_properties(test_engine_callbacks_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  py_test_modules(test_align_tool_deprecated MODULES test_align_tool_deprecated)
  set_tests_properties(test_align_tool_deprecated PROPERTIES TIMEOUT 20)
  py_test_modules(test_pass_recompute_deprecated MODULES
                  test_pass_recompute_deprecated)
  set_tests_properties(test_pass_recompute_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)
  py_test_modules(test_while_op_completion_deprecated MODULES
                  test_while_op_completion_deprecated)
  py_test_modules(test_while_op_partition_deprecated MODULES
                  test_while_op_partition_deprecated)
  py_test_modules(test_pattern_deprecated MODULES test_pattern_deprecated)
  py_test_modules(test_pattern_match_deprecated MODULES
                  test_pattern_match_deprecated)
  py_test_modules(test_rule_based_tuner_deprecated MODULES
                  test_rule_based_tuner_deprecated)
  py_test_modules(test_shard_layer_api_deprecated MODULES
                  test_shard_layer_api_deprecated)
  # End of unittests WITH single card WITHOUT timeout

  py_test_modules(test_pass_grad_clip_deprecated MODULES
                  test_pass_grad_clip_deprecated)
  set_tests_properties(test_pass_grad_clip_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)

  py_test_modules(test_pass_gradient_merge_deprecated MODULES
                  test_pass_gradient_merge_deprecated)
  set_tests_properties(test_pass_gradient_merge_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)
  py_test_modules(test_amp_o2_pass_deprecated MODULES
                  test_amp_o2_pass_deprecated)
  set_tests_properties(test_amp_o2_pass_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)
  py_test_modules(test_pass_bf16_deprecated MODULES test_pass_bf16_deprecated)
  py_test_modules(test_pass_base_list_deprecated MODULES
                  test_pass_base_list_deprecated)
  set_tests_properties(test_pass_base_list_deprecated PROPERTIES TIMEOUT 40)
  # NOTE(zyl): unittests WITH single card and WITHOUT timeout
  py_test_modules(test_serialization_deprecated MODULES
                  test_serialization_deprecated)
  py_test_modules(test_process_mesh_deprecated MODULES
                  test_process_mesh_deprecated)
  py_test_modules(test_new_cost_model_deprecated MODULES
                  test_new_cost_model_deprecated)
  py_test_modules(test_lr_grad_clip_deprecated MODULES
                  test_lr_grad_clip_deprecated)
  py_test_modules(test_interface_deprecated MODULES test_interface_deprecated)
  py_test_modules(test_group_operators_deprecated MODULES
                  test_group_operators_deprecated)
  py_test_modules(test_fp16_assign_deprecated MODULES
                  test_fp16_assign_deprecated)
  py_test_modules(test_engine_save_load_deprecated MODULES
                  test_engine_save_load_deprecated)
  py_test_modules(test_engine_api_error_deprecated MODULES
                  test_engine_api_error_deprecated)
  py_test_modules(test_dist_split_deprecated MODULES test_dist_split_deprecated)
  py_test_modules(test_dist_slice_deprecated MODULES test_dist_slice_deprecated)
  py_test_modules(test_dist_shape_deprecated MODULES test_dist_shape_deprecated)
  py_test_modules(test_dist_saver_deprecated MODULES test_dist_saver_deprecated)
  py_test_modules(test_dist_reshape_deprecated MODULES
                  test_dist_reshape_deprecated)
  py_test_modules(test_dist_pnorm_deprecated MODULES test_dist_pnorm_deprecated)
  py_test_modules(test_dist_embedding_deprecated MODULES
                  test_dist_embedding_deprecated)
  py_test_modules(test_dist_op_cost_deprecated MODULES
                  test_dist_op_cost_deprecated)
  py_test_modules(test_cost_interface_deprecated MODULES
                  test_cost_interface_deprecated)
  py_test_modules(test_conditional_block_reshard_deprecated MODULES
                  test_conditional_block_reshard_deprecated)
  py_test_modules(test_base_cost_deprecated MODULES test_base_cost_deprecated)
  py_test_modules(test_auto_conditional_block_deprecated MODULES
                  test_auto_conditional_block_deprecated)
  py_test_modules(test_to_static_deprecated MODULES test_to_static_deprecated)
  py_test_modules(test_dist_attr_v2_deprecated MODULES
                  test_dist_attr_v2_deprecated)
  py_test_modules(test_dist_matmul_deprecated MODULES
                  test_dist_matmul_deprecated)
  py_test_modules(test_dist_assign_deprecated MODULES
                  test_dist_assign_deprecated)
  py_test_modules(test_dist_concat_deprecated MODULES
                  test_dist_concat_deprecated)
  py_test_modules(test_dist_context_deprecated MODULES
                  test_dist_context_deprecated)
  py_test_modules(test_pipeline_scheduler_zb_vpp_deprecated MODULES
                  test_pipeline_scheduler_zb_vpp_deprecated)
  set_tests_properties(test_pipeline_scheduler_zb_vpp_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  py_test_modules(test_mp_allreduce_matmul_grad_overlapping_deprecated MODULES
                  test_mp_allreduce_matmul_grad_overlapping_deprecated)
  set_tests_properties(test_mp_allreduce_matmul_grad_overlapping_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)

  py_test_modules(test_auto_parallel_amp_pass_deprecated MODULES
                  test_auto_parallel_amp_pass_deprecated)
  set_tests_properties(test_mp_allreduce_matmul_grad_overlapping_deprecated
                       PROPERTIES LABELS "RUN_TYPE=DIST" TIMEOUT 250)
  py_test_modules(test_auto_parallel_recompute_pass_deprecated MODULES
                  test_auto_parallel_recompute_pass_deprecated)
  set_tests_properties(test_auto_parallel_recompute_pass_deprecated
                       PROPERTIES LABELS "RUN_TYPE=DIST" TIMEOUT 250)
  py_test_modules(test_auto_parallel_sharding_pass_deprecated MODULES
                  test_auto_parallel_sharding_pass_deprecated)
  set_tests_properties(test_auto_parallel_sharding_pass_deprecated
                       PROPERTIES LABELS "RUN_TYPE=DIST" TIMEOUT 250)
  py_test_modules(test_auto_parallel_fp16_pass_deprecated MODULES
                  test_auto_parallel_fp16_pass_deprecated)
  set_tests_properties(test_auto_parallel_fp16_pass_deprecated
                       PROPERTIES LABELS "RUN_TYPE=DIST" TIMEOUT 250)
  py_test_modules(test_auto_parallel_gradient_merge_pass_deprecated MODULES
                  test_auto_parallel_gradient_merge_pass_deprecated)
  set_tests_properties(test_auto_parallel_gradient_merge_pass_deprecated
                       PROPERTIES LABELS "RUN_TYPE=DIST" TIMEOUT 250)
  py_test_modules(
    test_auto_parallel_data_parallel_optimization_pass_deprecated MODULES
    test_auto_parallel_data_parallel_optimization_pass_deprecated)
  set_tests_properties(
    test_auto_parallel_data_parallel_optimization_pass_deprecated
    PROPERTIES LABELS "RUN_TYPE=DIST" TIMEOUT 250)
  if(CUDA_VERSION GREATER_EQUAL 11.6)
    py_test_modules(
      test_auto_parallel_fused_linear_promotion_pass_deprecated MODULES
      test_auto_parallel_fused_linear_promotion_pass_deprecated)
    set_tests_properties(
      test_auto_parallel_fused_linear_promotion_pass_deprecated
      PROPERTIES LABELS "RUN_TYPE=DIST" TIMEOUT 250)
  endif()
endif()
set_pir_tests_properties()
