if(WITH_DISTRIBUTE AND WITH_GPU)

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
