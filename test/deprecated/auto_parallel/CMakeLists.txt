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

  py_test_modules(test_pass_grad_clip MODULES test_pass_grad_clip)
  set_tests_properties(test_pass_grad_clip
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)

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
  py_test_modules(test_pass_quantization MODULES test_pass_quantization)
  set_tests_properties(test_pass_quantization
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 60)
  py_test_modules(test_pass_bf16 MODULES test_pass_bf16)
  py_test_modules(test_pass_base_list MODULES test_pass_base_list)
  set_tests_properties(test_pass_base_list PROPERTIES TIMEOUT 40)
  py_test_modules(test_fuse_adamw_pass MODULES test_fuse_adamw_pass)
  set_tests_properties(test_fuse_adamw_pass PROPERTIES TIMEOUT 20)
  py_test_modules(test_fused_linear_pass MODULES test_fused_linear_pass)
  set_tests_properties(test_fused_linear_pass PROPERTIES TIMEOUT 40)

endif()

set_pir_tests_properties()
