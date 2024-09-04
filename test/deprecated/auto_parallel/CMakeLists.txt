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

endif()

set_pir_tests_properties()
