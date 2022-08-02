if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet MODULES test_fleet ENVS
                  "PADDLE_DIST_UT_PORT=20057;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_ascend_utils MODULES test_fleet_ascend_utils ENVS
                  "PADDLE_DIST_UT_PORT=20059;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_ascend_utils PROPERTIES TIMEOUT "120")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_base_4 MODULES test_fleet_base_4 ENVS
                  "PADDLE_DIST_UT_PORT=20061;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_base_4 PROPERTIES TIMEOUT "120")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_base_single MODULES test_fleet_base_single ENVS
                  "PADDLE_DIST_UT_PORT=20063;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_base_single PROPERTIES TIMEOUT "120")
endif()
if((WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_elastic_collective MODULES test_fleet_elastic_collective ENVS
    "PADDLE_DIST_UT_PORT=20065;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_elastic_collective PROPERTIES TIMEOUT "350")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_elastic_init MODULES test_fleet_elastic_init ENVS
                  "PADDLE_DIST_UT_PORT=20067;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_elastic_init PROPERTIES TIMEOUT "300")
endif()
if((WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_elastic_manager MODULES test_fleet_elastic_manager
                  ENVS "PADDLE_DIST_UT_PORT=20069;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_elastic_manager PROPERTIES TIMEOUT "350")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_exe_dist_model_run MODULES test_fleet_exe_dist_model_run ENVS
    "PADDLE_DIST_UT_PORT=20071;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_exe_dist_model_run PROPERTIES TIMEOUT "120")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_exe_dist_model_tensor MODULES test_fleet_exe_dist_model_tensor
    ENVS "PADDLE_DIST_UT_PORT=20073;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_exe_dist_model_tensor PROPERTIES TIMEOUT
                                                                   "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_executor MODULES test_fleet_executor ENVS
                  "PADDLE_DIST_UT_PORT=20075;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_executor PROPERTIES TIMEOUT "200")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_executor_multi_devices MODULES test_fleet_executor_multi_devices
    ENVS "PADDLE_DIST_UT_PORT=20077;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_executor_multi_devices PROPERTIES TIMEOUT
                                                                    "200")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_executor_origin_scheduler MODULES
    test_fleet_executor_origin_scheduler ENVS
    "PADDLE_DIST_UT_PORT=20079;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_executor_origin_scheduler PROPERTIES TIMEOUT
                                                                       "300")
endif()
if((WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_executor_task_node MODULES test_fleet_executor_task_node ENVS
    "PADDLE_DIST_UT_PORT=20081;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_executor_task_node PROPERTIES TIMEOUT "350")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_executor_with_task_nodes MODULES
    test_fleet_executor_with_task_nodes ENVS
    "PADDLE_DIST_UT_PORT=20083;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_executor_with_task_nodes PROPERTIES TIMEOUT
                                                                      "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_gradient_scale MODULES test_fleet_gradient_scale
                  ENVS "PADDLE_DIST_UT_PORT=20085;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_gradient_scale PROPERTIES TIMEOUT "300")
endif()
if((WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_metric MODULES test_fleet_metric ENVS
                  "PADDLE_DIST_UT_PORT=20087;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_metric PROPERTIES TIMEOUT "350")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_nocvm_1 MODULES test_fleet_nocvm_1 ENVS
                  "PADDLE_DIST_UT_PORT=20089;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_nocvm_1 PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_ps MODULES test_fleet_ps ENVS
                  "PADDLE_DIST_UT_PORT=20091;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_ps PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_rolemaker MODULES test_fleet_rolemaker ENVS
                  "PADDLE_DIST_UT_PORT=20093;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_rolemaker PROPERTIES TIMEOUT "120")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_rolemaker_2 MODULES test_fleet_rolemaker_2 ENVS
                  "PADDLE_DIST_UT_PORT=20095;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_rolemaker_2 PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_rolemaker_3 MODULES test_fleet_rolemaker_3 ENVS
                  "PADDLE_DIST_UT_PORT=20097;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_rolemaker_3 PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_rolemaker_4 MODULES test_fleet_rolemaker_4 ENVS
                  "PADDLE_DIST_UT_PORT=20099;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_rolemaker_4 PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_rolemaker_init MODULES test_fleet_rolemaker_init
                  ENVS "PADDLE_DIST_UT_PORT=20101;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_rolemaker_init PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_rolemaker_new MODULES test_fleet_rolemaker_new
                  ENVS "PADDLE_DIST_UT_PORT=20103;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_rolemaker_new PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_runtime MODULES test_fleet_runtime ENVS
                  "PADDLE_DIST_UT_PORT=20105;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_runtime PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_unitaccessor MODULES test_fleet_unitaccessor ENVS
                  "PADDLE_DIST_UT_PORT=20107;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_unitaccessor PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_util MODULES test_fleet_util ENVS
                  "PADDLE_DIST_UT_PORT=20109;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_util PROPERTIES TIMEOUT "120")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_utils MODULES test_fleet_utils ENVS
                  "PADDLE_DIST_UT_PORT=20111;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_utils PROPERTIES TIMEOUT "120")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_base MODULES test_fleet_base ENVS
                  "PADDLE_DIST_UT_PORT=20113;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_base PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_base_2 MODULES test_fleet_base_2 ENVS
                  "PADDLE_DIST_UT_PORT=20115;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_base_2 PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_base_3 MODULES test_fleet_base_3 ENVS
                  "PADDLE_DIST_UT_PORT=20117;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_base_3 PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_amp_init MODULES test_fleet_amp_init ENVS
                  "PADDLE_DIST_UT_PORT=20119;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_amp_init PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_fp16_allreduce_meta_optimizer MODULES
    test_fleet_fp16_allreduce_meta_optimizer ENVS
    "PADDLE_DIST_UT_PORT=20121;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_fp16_allreduce_meta_optimizer
                       PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_private_function MODULES test_fleet_private_function ENVS
    "PADDLE_DIST_UT_PORT=20123;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_private_function PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_meta_optimizer_base MODULES test_fleet_meta_optimizer_base ENVS
    "PADDLE_DIST_UT_PORT=20125;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_meta_optimizer_base PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_distributed_strategy MODULES test_fleet_distributed_strategy
    ENVS "PADDLE_DIST_UT_PORT=20127;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_distributed_strategy PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_static_mp_layers MODULES test_fleet_static_mp_layers ENVS
    "PADDLE_DIST_UT_PORT=20129;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_static_mp_layers PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_amp_meta_optimizer MODULES test_fleet_amp_meta_optimizer ENVS
    "PADDLE_DIST_UT_PORT=20131;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_amp_meta_optimizer PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_gradient_merge_meta_optimizer MODULES
    test_fleet_gradient_merge_meta_optimizer ENVS
    "PADDLE_DIST_UT_PORT=20133;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_gradient_merge_meta_optimizer
                       PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_graph_executor MODULES test_fleet_graph_executor
                  ENVS "PADDLE_DIST_UT_PORT=20135;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_graph_executor PROPERTIES TIMEOUT "400")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_hybrid_meta_optimizer MODULES test_fleet_hybrid_meta_optimizer
    ENVS "PADDLE_DIST_UT_PORT=20137;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_hybrid_meta_optimizer PROPERTIES TIMEOUT
                                                                   "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_recompute_meta_optimizer MODULES
    test_fleet_recompute_meta_optimizer ENVS
    "PADDLE_DIST_UT_PORT=20139;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_recompute_meta_optimizer PROPERTIES TIMEOUT
                                                                      "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_sharding_meta_optimizer MODULES
    test_fleet_sharding_meta_optimizer ENVS
    "PADDLE_DIST_UT_PORT=20141;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_sharding_meta_optimizer PROPERTIES TIMEOUT
                                                                     "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_lamb_meta_optimizer MODULES test_fleet_lamb_meta_optimizer ENVS
    "PADDLE_DIST_UT_PORT=20143;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_lamb_meta_optimizer PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_lars_meta_optimizer MODULES test_fleet_lars_meta_optimizer ENVS
    "PADDLE_DIST_UT_PORT=20145;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_lars_meta_optimizer PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_localsgd_meta_optimizer MODULES
    test_fleet_localsgd_meta_optimizer ENVS
    "PADDLE_DIST_UT_PORT=20147;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_localsgd_meta_optimizer PROPERTIES TIMEOUT
                                                                     "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_dgc_meta_optimizer MODULES test_fleet_dgc_meta_optimizer ENVS
    "PADDLE_DIST_UT_PORT=20149;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_dgc_meta_optimizer PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_checkpoint MODULES test_fleet_checkpoint ENVS
                  "PADDLE_DIST_UT_PORT=20151;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_checkpoint PROPERTIES TIMEOUT "400")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_fleet_graph_execution_meta_optimizer
    START_BASH
    dist_test.sh
    LABELS
    "RUN_TYPE=EXCLUSIVE"
    ENVS
    "PADDLE_DIST_UT_PORT=20153;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_graph_execution_meta_optimizer
                       PROPERTIES TIMEOUT "350")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_fleet_pipeline_meta_optimizer
    START_BASH
    dist_test.sh
    LABELS
    "RUN_TYPE=EXCLUSIVE"
    ENVS
    "PADDLE_DIST_UT_PORT=20155;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_pipeline_meta_optimizer PROPERTIES TIMEOUT
                                                                     "350")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_fleet_pipeline_meta_optimizer_with_recompute
    START_BASH
    dist_test.sh
    LABELS
    "RUN_TYPE=EXCLUSIVE"
    ENVS
    "PADDLE_DIST_UT_PORT=20157;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_pipeline_meta_optimizer_with_recompute
                       PROPERTIES TIMEOUT "350")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_fleet_raw_program_meta_optimizer
    START_BASH
    dist_test.sh
    LABELS
    "RUN_TYPE=EXCLUSIVE"
    ENVS
    "PADDLE_DIST_UT_PORT=20159;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_raw_program_meta_optimizer PROPERTIES TIMEOUT
                                                                        "350")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_fleetrun
    START_BASH
    test_fleetrun.sh
    LABELS
    "RUN_TYPE=DIST"
    ENVS
    "PADDLE_DIST_UT_PORT=20161;http_proxy=;https_proxy=;TEST_TIMEOUT=350")
  set_tests_properties(test_fleetrun PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_fleet_launch_nproc
    START_BASH
    test_fleet_launch_nproc.sh
    LABELS
    "RUN_TYPE=DIST"
    ENVS
    "PADDLE_DIST_UT_PORT=20163;http_proxy=;https_proxy=;TEST_TIMEOUT=350")
  set_tests_properties(test_fleet_launch_nproc PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_fleet_run_random_port
    START_BASH
    test_fleet_run_random_port.sh
    LABELS
    "RUN_TYPE=DIST"
    ENVS
    "PADDLE_DIST_UT_PORT=20165;http_proxy=;https_proxy=;TEST_TIMEOUT=350")
  set_tests_properties(test_fleet_run_random_port PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_fleet_launch_async
    START_BASH
    test_fleet_launch_async.sh
    LABELS
    "RUN_TYPE=DIST"
    ENVS
    "PADDLE_DIST_UT_PORT=20167;http_proxy=;https_proxy=;TEST_TIMEOUT=350")
  set_tests_properties(test_fleet_launch_async PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_fleet_launch_cloud
    START_BASH
    test_fleet_launch_cloud.sh
    LABELS
    "RUN_TYPE=DIST"
    ENVS
    "PADDLE_DIST_UT_PORT=20169;http_proxy=;https_proxy=;TEST_TIMEOUT=350")
  set_tests_properties(test_fleet_launch_cloud PROPERTIES TIMEOUT "300")
endif()
if((WITH_ROCM OR WITH_GPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_fleet_launch_ps
    START_BASH
    test_fleet_launch_ps.sh
    LABELS
    "RUN_TYPE=EXCLUSIVE"
    ENVS
    "PADDLE_DIST_UT_PORT=20171;http_proxy=;https_proxy=;TEST_TIMEOUT=350")
  set_tests_properties(test_fleet_launch_ps PROPERTIES TIMEOUT "300")
endif()
