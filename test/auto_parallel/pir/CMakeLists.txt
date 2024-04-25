if(WITH_DISTRIBUTE AND WITH_GPU)
  py_test_modules(test_to_static_pir_program MODULES test_to_static_pir_program)
  set_tests_properties(test_to_static_pir_program
                       PROPERTIES ENVIRONMENT "FLAGS_enable_pir_api=1")
  py_test_modules(test_ir_dist_attr MODULES test_ir_dist_attr ENVS
                  FLAGS_enable_pir_api=1)
  py_test_modules(test_static_pir_program MODULES test_static_pir_program)
  py_test_modules(test_pir_elementwise_spmd MODULES test_elementwise_spmd_rule
                  ENVS FLAGS_enable_pir_api=1)
  py_test_modules(test_pir_relu_spmd MODULES test_relu_spmd_rule ENVS
                  FLAGS_enable_pir_api=1)
  py_test_modules(test_pir_mse_spmd MODULES test_mse_spmd_rule ENVS
                  FLAGS_enable_pir_api=1)
  py_test_modules(test_mlp MODULES test_mlp ENVS FLAGS_enable_pir_api=1)
  py_test_modules(
    test_semi_auto_parallel_dist_to_static_pir MODULES
    test_semi_auto_parallel_dist_to_static_pir ENVS FLAGS_enable_pir_api=1)
  py_test_modules(test_reshard MODULES test_reshard ENVS FLAGS_enable_pir_api=1)
  py_test_modules(test_learning_rate MODULES test_learning_rate ENVS
                  FLAGS_enable_pir_api=1)
  set_tests_properties(test_mlp PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT
                                           60)
  set_tests_properties(test_semi_auto_parallel_dist_to_static_pir
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 30)
endif()
