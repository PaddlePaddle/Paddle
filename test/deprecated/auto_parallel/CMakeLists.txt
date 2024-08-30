# file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
# string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

if(WITH_DISTRIBUTE AND WITH_GPU)
  py_test_modules(test_mp_allreduce_matmul_grad_overlapping_deprecated MODULES
                  test_mp_allreduce_matmul_grad_overlapping_deprecated)
  set_tests_properties(test_mp_allreduce_matmul_grad_overlapping_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)
  py_test_modules(test_auto_parallel_relaunch_deprecated MODULES
                  test_auto_parallel_relaunch_deprecated)
  set_tests_properties(test_auto_parallel_relaunch_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)
  py_test_modules(test_pass_1F1B_deprecated MODULES test_pass_1F1B_deprecated)
  set_tests_properties(test_pass_1F1B_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)
  set_tests_properties(
    test_pass_1F1B_deprecated
    PROPERTIES ENVIRONMENT "FLAGS_new_executor_micro_batching=False")
  py_test_modules(test_amp_o2_pass_deprecated MODULES
                  test_amp_o2_pass_deprecated)
  set_tests_properties(test_amp_o2_pass_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)
  py_test_modules(test_pass_amp_deprecated MODULES test_pass_amp_deprecated)
  set_tests_properties(test_pass_amp_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 80)
  py_test_modules(test_relaunch_with_planner_deprecated MODULES
                  test_relaunch_with_planner_deprecated)
  set_tests_properties(test_relaunch_with_planner_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)
  py_test_modules(test_pass_grad_clip_deprecated MODULES
                  test_pass_grad_clip_deprecated)
  set_tests_properties(test_pass_grad_clip_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)
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
  py_test_modules(test_pass_generation_pipeline_deprecated MODULES
                  test_pass_generation_pipeline_deprecated)
  set_tests_properties(test_pass_generation_pipeline_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)
  set_tests_properties(
    test_pass_generation_pipeline_deprecated
    PROPERTIES ENVIRONMENT "FLAGS_new_executor_micro_batching=False")
  py_test_modules(test_gpt_with_prim_deprecated MODULES
                  test_gpt_with_prim_deprecated)
  set_tests_properties(test_gpt_with_prim_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 200)
  py_test_modules(test_pass_gradient_merge_deprecated MODULES
                  test_pass_gradient_merge_deprecated)
  set_tests_properties(test_pass_gradient_merge_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)
  py_test_modules(test_optimization_tuner_api_deprecated MODULES
                  test_optimization_tuner_api_deprecated)
  set_tests_properties(test_optimization_tuner_api_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)
endif()

set_pir_tests_properties()
