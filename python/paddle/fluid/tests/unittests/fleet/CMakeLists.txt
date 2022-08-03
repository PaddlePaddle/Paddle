set(LOCAL_ALL_ARCH ON)
set(LOCAL_ALL_PLAT ON)
if(LOCAL_ALL_ARCH AND LOCAL_ALL_PLAT)
  py_test_modules(test_ema_fleet MODULES test_ema_fleet ENVS
                  "PADDLE_DIST_UT_PORT=20173;http_proxy=;https_proxy=")
  set_tests_properties(test_ema_fleet PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND LOCAL_ALL_PLAT)
  py_test_modules(test_fleet MODULES test_fleet ENVS
                  "PADDLE_DIST_UT_PORT=20175;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet PROPERTIES TIMEOUT "120")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_amp_init MODULES test_fleet_amp_init ENVS
    "PADDLE_DIST_UT_PORT=20177;http_proxy=;https_proxy=;CUDA_VISIBLE_DEVICES=0")
  set_tests_properties(test_fleet_amp_init PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_amp_meta_optimizer MODULES test_fleet_amp_meta_optimizer ENVS
    "PADDLE_DIST_UT_PORT=20179;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_amp_meta_optimizer PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND LOCAL_ALL_PLAT)
  py_test_modules(test_fleet_ascend_utils MODULES test_fleet_ascend_utils ENVS
                  "PADDLE_DIST_UT_PORT=20181;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_ascend_utils PROPERTIES TIMEOUT "120")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_base MODULES test_fleet_base ENVS
                  "PADDLE_DIST_UT_PORT=20183;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_base PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_base_2 MODULES test_fleet_base_2 ENVS
                  "PADDLE_DIST_UT_PORT=20185;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_base_2 PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_base_3 MODULES test_fleet_base_3 ENVS
                  "PADDLE_DIST_UT_PORT=20187;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_base_3 PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_base_4 MODULES test_fleet_base_4 ENVS
                  "PADDLE_DIST_UT_PORT=20189;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_base_4 PROPERTIES TIMEOUT "120")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_base_single MODULES test_fleet_base_single ENVS
    "PADDLE_DIST_UT_PORT=20191;http_proxy=;https_proxy=;CUDA_VISIBLE_DEVICES=0")
  set_tests_properties(test_fleet_base_single PROPERTIES TIMEOUT "120")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_checkpoint MODULES test_fleet_checkpoint ENVS
                  "PADDLE_DIST_UT_PORT=20193;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_checkpoint PROPERTIES TIMEOUT "500")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_dgc_meta_optimizer MODULES test_fleet_dgc_meta_optimizer ENVS
    "PADDLE_DIST_UT_PORT=20195;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_dgc_meta_optimizer PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_distributed_strategy MODULES test_fleet_distributed_strategy
    ENVS "PADDLE_DIST_UT_PORT=20197;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_distributed_strategy PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND LOCAL_ALL_PLAT)
  py_test_modules(
    test_fleet_elastic_collective MODULES test_fleet_elastic_collective ENVS
    "PADDLE_DIST_UT_PORT=20199;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_elastic_collective PROPERTIES TIMEOUT "350")
endif()
if(LOCAL_ALL_ARCH AND LOCAL_ALL_PLAT)
  py_test_modules(test_fleet_elastic_init MODULES test_fleet_elastic_init ENVS
                  "PADDLE_DIST_UT_PORT=20201;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_elastic_init PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND LOCAL_ALL_PLAT)
  py_test_modules(test_fleet_elastic_manager MODULES test_fleet_elastic_manager
                  ENVS "PADDLE_DIST_UT_PORT=20203;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_elastic_manager PROPERTIES TIMEOUT "350")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_exe_dist_model_run MODULES test_fleet_exe_dist_model_run ENVS
    "PADDLE_DIST_UT_PORT=20205;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_exe_dist_model_run PROPERTIES TIMEOUT "120")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_exe_dist_model_tensor MODULES test_fleet_exe_dist_model_tensor
    ENVS "PADDLE_DIST_UT_PORT=20207;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_exe_dist_model_tensor PROPERTIES TIMEOUT
                                                                   "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_executor MODULES test_fleet_executor ENVS
                  "PADDLE_DIST_UT_PORT=20209;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_executor PROPERTIES TIMEOUT "200")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_executor_multi_devices MODULES test_fleet_executor_multi_devices
    ENVS "PADDLE_DIST_UT_PORT=20211;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_executor_multi_devices PROPERTIES TIMEOUT
                                                                    "200")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_executor_origin_scheduler MODULES
    test_fleet_executor_origin_scheduler ENVS
    "PADDLE_DIST_UT_PORT=20213;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_executor_origin_scheduler PROPERTIES TIMEOUT
                                                                       "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_executor_task_node MODULES test_fleet_executor_task_node ENVS
    "PADDLE_DIST_UT_PORT=20215;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_executor_task_node PROPERTIES TIMEOUT "350")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_executor_with_task_nodes MODULES
    test_fleet_executor_with_task_nodes ENVS
    "PADDLE_DIST_UT_PORT=20217;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_executor_with_task_nodes PROPERTIES TIMEOUT
                                                                      "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_fp16_allreduce_meta_optimizer MODULES
    test_fleet_fp16_allreduce_meta_optimizer ENVS
    "PADDLE_DIST_UT_PORT=20219;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_fp16_allreduce_meta_optimizer
                       PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_gradient_merge_meta_optimizer MODULES
    test_fleet_gradient_merge_meta_optimizer ENVS
    "PADDLE_DIST_UT_PORT=20221;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_gradient_merge_meta_optimizer
                       PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_gradient_scale MODULES test_fleet_gradient_scale
                  ENVS "PADDLE_DIST_UT_PORT=20223;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_gradient_scale PROPERTIES TIMEOUT "300")
endif()
if((WITH_GPU OR WITH_XPU) AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_fleet_graph_execution_meta_optimizer
    START_BASH
    dist_test.sh
    LABELS
    "RUN_TYPE=EXCLUSIVE"
    ENVS
    "PADDLE_DIST_UT_PORT=20225;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_graph_execution_meta_optimizer
                       PROPERTIES TIMEOUT "350")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_graph_executor MODULES test_fleet_graph_executor
                  ENVS "PADDLE_DIST_UT_PORT=20227;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_graph_executor PROPERTIES TIMEOUT "400")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_hybrid_meta_optimizer MODULES test_fleet_hybrid_meta_optimizer
    ENVS "PADDLE_DIST_UT_PORT=20229;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_hybrid_meta_optimizer PROPERTIES TIMEOUT
                                                                   "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_lamb_meta_optimizer MODULES test_fleet_lamb_meta_optimizer ENVS
    "PADDLE_DIST_UT_PORT=20231;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_lamb_meta_optimizer PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_lars_meta_optimizer MODULES test_fleet_lars_meta_optimizer ENVS
    "PADDLE_DIST_UT_PORT=20233;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_lars_meta_optimizer PROPERTIES TIMEOUT "300")
endif()
if((WITH_GPU
    OR WITH_XPU
    OR WITH_ASCEND
    OR WITH_ASCEND_CL
   )
   AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_fleet_launch_async
    START_BASH
    test_fleet_launch_async.sh
    LABELS
    "RUN_TYPE=DIST"
    ENVS
    "PADDLE_DIST_UT_PORT=20235;http_proxy=;https_proxy=;TEST_TIMEOUT=350")
  set_tests_properties(test_fleet_launch_async PROPERTIES TIMEOUT "300")
endif()
if((WITH_GPU
    OR WITH_XPU
    OR WITH_ASCEND
    OR WITH_ASCEND_CL
   )
   AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_fleet_launch_cloud
    START_BASH
    test_fleet_launch_cloud.sh
    LABELS
    "RUN_TYPE=DIST"
    ENVS
    "PADDLE_DIST_UT_PORT=20237;http_proxy=;https_proxy=;TEST_TIMEOUT=350")
  set_tests_properties(test_fleet_launch_cloud PROPERTIES TIMEOUT "300")
endif()
if((WITH_GPU
    OR WITH_XPU
    OR WITH_ASCEND_CL
    OR WITH_ASCEND
   )
   AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_fleet_launch_nproc
    START_BASH
    test_fleet_launch_nproc.sh
    LABELS
    "RUN_TYPE=DIST"
    ENVS
    "PADDLE_DIST_UT_PORT=20239;http_proxy=;https_proxy=;TEST_TIMEOUT=350")
  set_tests_properties(test_fleet_launch_nproc PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_fleet_launch_ps
    START_BASH
    test_fleet_launch_ps.sh
    LABELS
    "RUN_TYPE=EXCLUSIVE"
    ENVS
    "PADDLE_DIST_UT_PORT=20241;http_proxy=;https_proxy=;TEST_TIMEOUT=350")
  set_tests_properties(test_fleet_launch_ps PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_localsgd_meta_optimizer MODULES
    test_fleet_localsgd_meta_optimizer ENVS
    "PADDLE_DIST_UT_PORT=20243;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_localsgd_meta_optimizer PROPERTIES TIMEOUT
                                                                     "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_meta_optimizer_base MODULES test_fleet_meta_optimizer_base ENVS
    "PADDLE_DIST_UT_PORT=20245;http_proxy=;https_proxy=;CUDA_VISIBLE_DEVICES=0")
  set_tests_properties(test_fleet_meta_optimizer_base PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_metric MODULES test_fleet_metric ENVS
                  "PADDLE_DIST_UT_PORT=20247;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_metric PROPERTIES TIMEOUT "350")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_nocvm_1 MODULES test_fleet_nocvm_1 ENVS
                  "PADDLE_DIST_UT_PORT=20249;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_nocvm_1 PROPERTIES TIMEOUT "300")
endif()
if((WITH_GPU
    OR WITH_XPU
    OR WITH_ASCEND
    OR WITH_ASCEND_CL
   )
   AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_fleet_pipeline_meta_optimizer
    START_BASH
    dist_test.sh
    LABELS
    "RUN_TYPE=EXCLUSIVE"
    ENVS
    "PADDLE_DIST_UT_PORT=20251;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_pipeline_meta_optimizer PROPERTIES TIMEOUT
                                                                     "350")
endif()
if((WITH_GPU
    OR WITH_XPU
    OR WITH_ASCEND
    OR WITH_ASCEND_CL
   )
   AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_fleet_pipeline_meta_optimizer_with_recompute
    START_BASH
    dist_test.sh
    LABELS
    "RUN_TYPE=EXCLUSIVE"
    ENVS
    "PADDLE_DIST_UT_PORT=20253;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_pipeline_meta_optimizer_with_recompute
                       PROPERTIES TIMEOUT "350")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_private_function MODULES test_fleet_private_function ENVS
    "PADDLE_DIST_UT_PORT=20255;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_private_function PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_ps MODULES test_fleet_ps ENVS
                  "PADDLE_DIST_UT_PORT=20257;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_ps PROPERTIES TIMEOUT "300")
endif()
if((WITH_GPU
    OR WITH_XPU
    OR WITH_ASCEND
    OR WITH_ASCEND_CL
   )
   AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_fleet_raw_program_meta_optimizer
    START_BASH
    dist_test.sh
    LABELS
    "RUN_TYPE=EXCLUSIVE"
    ENVS
    "PADDLE_DIST_UT_PORT=20259;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_raw_program_meta_optimizer PROPERTIES TIMEOUT
                                                                        "350")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_recompute_meta_optimizer MODULES
    test_fleet_recompute_meta_optimizer ENVS
    "PADDLE_DIST_UT_PORT=20261;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_recompute_meta_optimizer PROPERTIES TIMEOUT
                                                                      "300")
endif()
if(LOCAL_ALL_ARCH AND LOCAL_ALL_PLAT)
  py_test_modules(test_fleet_rolemaker MODULES test_fleet_rolemaker ENVS
                  "PADDLE_DIST_UT_PORT=20263;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_rolemaker PROPERTIES TIMEOUT "120")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_rolemaker_2 MODULES test_fleet_rolemaker_2 ENVS
                  "PADDLE_DIST_UT_PORT=20265;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_rolemaker_2 PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_rolemaker_3 MODULES test_fleet_rolemaker_3 ENVS
                  "PADDLE_DIST_UT_PORT=20267;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_rolemaker_3 PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_rolemaker_4 MODULES test_fleet_rolemaker_4 ENVS
                  "PADDLE_DIST_UT_PORT=20269;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_rolemaker_4 PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND LOCAL_ALL_PLAT)
  py_test_modules(test_fleet_rolemaker_init MODULES test_fleet_rolemaker_init
                  ENVS "PADDLE_DIST_UT_PORT=20271;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_rolemaker_init PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_fleet_rolemaker_new MODULES test_fleet_rolemaker_new
                  ENVS "PADDLE_DIST_UT_PORT=20273;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_rolemaker_new PROPERTIES TIMEOUT "300")
endif()
if((WITH_GPU
    OR WITH_XPU
    OR WITH_ASCEND
    OR WITH_ASCEND_CL
   )
   AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_fleet_run_random_port
    START_BASH
    test_fleet_run_random_port.sh
    LABELS
    "RUN_TYPE=DIST"
    ENVS
    "PADDLE_DIST_UT_PORT=20275;http_proxy=;https_proxy=;TEST_TIMEOUT=350")
  set_tests_properties(test_fleet_run_random_port PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND LOCAL_ALL_PLAT)
  py_test_modules(test_fleet_runtime MODULES test_fleet_runtime ENVS
                  "PADDLE_DIST_UT_PORT=20277;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_runtime PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_sharding_meta_optimizer MODULES
    test_fleet_sharding_meta_optimizer ENVS
    "PADDLE_DIST_UT_PORT=20279;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_sharding_meta_optimizer PROPERTIES TIMEOUT
                                                                     "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_fleet_static_mp_layers MODULES test_fleet_static_mp_layers ENVS
    "PADDLE_DIST_UT_PORT=20281;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_static_mp_layers PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND LOCAL_ALL_PLAT)
  py_test_modules(test_fleet_unitaccessor MODULES test_fleet_unitaccessor ENVS
                  "PADDLE_DIST_UT_PORT=20283;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_unitaccessor PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND LOCAL_ALL_PLAT)
  py_test_modules(test_fleet_util MODULES test_fleet_util ENVS
                  "PADDLE_DIST_UT_PORT=20285;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_util PROPERTIES TIMEOUT "120")
endif()
if(LOCAL_ALL_ARCH AND LOCAL_ALL_PLAT)
  py_test_modules(test_fleet_utils MODULES test_fleet_utils ENVS
                  "PADDLE_DIST_UT_PORT=20287;http_proxy=;https_proxy=")
  set_tests_properties(test_fleet_utils PROPERTIES TIMEOUT "120")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_fleetrun
    START_BASH
    test_fleetrun.sh
    LABELS
    "RUN_TYPE=DIST"
    ENVS
    "PADDLE_DIST_UT_PORT=20289;http_proxy=;https_proxy=;TEST_TIMEOUT=350")
  set_tests_properties(test_fleetrun PROPERTIES TIMEOUT "300")
endif()
if((WITH_GPU
    OR WITH_XPU
    OR WITH_ASCEND_CL
    OR WITH_ASCEND
   )
   AND LOCAL_ALL_PLAT)
  bash_test_modules(
    test_gen_nccl_id_op
    START_BASH
    dist_test.sh
    LABELS
    "RUN_TYPE=EXCLUSIVE"
    ENVS
    "PADDLE_DIST_UT_PORT=20291;http_proxy=;https_proxy=")
  set_tests_properties(test_gen_nccl_id_op PROPERTIES TIMEOUT "120")
endif()
