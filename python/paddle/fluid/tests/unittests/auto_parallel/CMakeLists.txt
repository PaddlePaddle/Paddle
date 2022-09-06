# file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
# string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
if(WITH_DISTRIBUTE AND WITH_GPU)
  py_test_modules(test_auto_parallel_relaunch MODULES
                  test_auto_parallel_relaunch ENVS ${dist_ENVS})
  set_tests_properties(test_auto_parallel_relaunch
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)

  py_test_modules(test_relaunch_with_planner MODULES test_relaunch_with_planner
                  ENVS ${dist_ENVS})
  set_tests_properties(test_relaunch_with_planner
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)

  py_test_modules(test_relaunch_with_gpt_planner MODULES
                  test_relaunch_with_gpt_planner ENVS ${dist_ENVS})
  set_tests_properties(test_relaunch_with_gpt_planner
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 240)

  py_test_modules(test_engine_api MODULES test_engine_api ENVS ${dist_ENVS})
  set_tests_properties(test_engine_api PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE"
                                                  TIMEOUT 80)

  py_test_modules(test_engine_api_dp MODULES test_engine_api_dp ENVS
                  ${dist_ENVS})
  set_tests_properties(test_engine_api_dp
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 80)

  py_test_modules(test_optimization_tuner_api MODULES
                  test_optimization_tuner_api ENVS ${dist_ENVS})
  set_tests_properties(test_optimization_tuner_api
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 80)

  py_test_modules(test_converter MODULES test_converter ENVS ${dist_ENVS})
  set_tests_properties(test_converter PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE"
                                                 TIMEOUT 50)
  py_test_modules(test_high_order_grad MODULES test_high_order_grad ENVS
                  ${dist_ENVS})
  set_tests_properties(test_high_order_grad
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)
  py_test_modules(test_grad_clip MODULES test_grad_clip ENVS ${dist_ENVS})
  set_tests_properties(test_grad_clip PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE"
                                                 TIMEOUT 50)

  py_test_modules(test_while_op_completion MODULES test_while_op_completion
                  ENVS ${dist_ENVS})
  py_test_modules(test_while_op_partition MODULES test_while_op_partition ENVS
                  ${dist_ENVS})
  py_test_modules(test_tunable_variable MODULES test_tunable_variable ENVS
                  ${dist_ENVS})
  py_test_modules(test_tunable_space MODULES test_tunable_space ENVS
                  ${dist_ENVS})
  py_test_modules(test_recorder MODULES test_recorder ENVS ${dist_ENVS})
  py_test_modules(test_trial MODULES test_trial ENVS ${dist_ENVS})
  py_test_modules(test_new_cost_model MODULES test_new_cost_model ENVS
                  ${dist_ENVS})
  py_test_modules(test_dist_reshape MODULES test_dist_reshape ENVS ${dist_ENVS})
  py_test_modules(test_dist_pnorm MODULES test_dist_pnorm ENVS ${dist_ENVS})
  py_test_modules(test_dist_embedding MODULES test_dist_embedding ENVS
                  ${dist_ENVS})
  py_test_modules(test_dist_slice MODULES test_dist_slice ENVS ${dist_ENVS})
  py_test_modules(test_cluster MODULES test_cluster ENVS ${dist_ENVS})
  py_test_modules(test_comm_cost MODULES test_comm_cost ENVS ${dist_ENVS})
  py_test_modules(test_comp_cost MODULES test_comp_cost ENVS ${dist_ENVS})
  py_test_modules(test_base_cost MODULES test_base_cost ENVS ${dist_ENVS})
  py_test_modules(test_dist_context MODULES test_dist_context ENVS ${dist_ENVS})
  py_test_modules(test_prim_dist_op MODULES test_prim_dist_op ENVS ${dist_ENVS})
  py_test_modules(test_to_static MODULES test_to_static ENVS ${dist_ENVS})
  py_test_modules(test_dist_op_cost MODULES test_dist_op_cost ENVS ${dist_ENVS})
  py_test_modules(test_cluster_v2 MODULES test_cluster_v2)
  py_test_modules(test_process_mesh_v2 MODULES test_process_mesh_v2)
  py_test_modules(test_dist_attr_v2 MODULES test_dist_attr_v2)
  py_test_modules(test_lr_grad_clip MODULES test_lr_grad_clip)
  py_test_modules(test_quantization MODULES test_quantization)
  py_test_modules(test_dist_matmul MODULES test_dist_matmul)
endif()
