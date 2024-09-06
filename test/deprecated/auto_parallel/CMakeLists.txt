# file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
# string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

# add_subdirectory(spmd_rules)
# add_subdirectory(hybrid_strategy)
# add_subdirectory(custom_op)
# add_subdirectory(pir)

if(WITH_DISTRIBUTE AND WITH_GPU)

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

endif()

set_pir_tests_properties()
