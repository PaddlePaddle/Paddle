file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

if((NOT WITH_GPU) AND (NOT WITH_XPU))
  list(REMOVE_ITEM TEST_OPS "test_dist_fuse_adam_pass_deprecated")
  list(REMOVE_ITEM TEST_OPS "test_dist_fuse_bn_act_pass_deprecated")
  list(REMOVE_ITEM TEST_OPS "test_dist_fuse_bn_add_act_pass_deprecated")
  list(REMOVE_ITEM TEST_OPS "test_dist_fuse_momentum_pass_deprecated")
  list(REMOVE_ITEM TEST_OPS
       "test_dist_fuse_relu_depthwise_conv_pass_deprecated")
  list(REMOVE_ITEM TEST_OPS "test_dist_fuse_sgd_pass_deprecated")
  list(REMOVE_ITEM TEST_OPS "test_dist_inplace_addto_pass_deprecated")
  list(REMOVE_ITEM TEST_OPS "test_auto_parallel_amp_pass_deprecated")
  list(REMOVE_ITEM TEST_OPS "test_auto_parallel_recompute_pass_deprecated")
  list(REMOVE_ITEM TEST_OPS "test_auto_parallel_sharding_pass_deprecated")
  list(REMOVE_ITEM TEST_OPS "test_auto_parallel_fp16_pass_deprecated")
  list(REMOVE_ITEM TEST_OPS "test_auto_parallel_gradient_merge_pass_deprecated")
  list(REMOVE_ITEM TEST_OPS
       "test_auto_parallel_data_parallel_optimization_pass_deprecated")
endif()

if(NOT ((WITH_GPU) AND (CUDA_VERSION GREATER_EQUAL 11.6)))
  list(REMOVE_ITEM TEST_OPS
       test_auto_parallel_fused_linear_promotion_pass_deprecated)
endif()

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP} ENVS "NVIDIA_TF32_OVERRIDE=0")
  list(APPEND DIST_TEST_OPS ${TEST_OP})
  set_tests_properties(${TEST_OP} PROPERTIES TIMEOUT 250)
  set_tests_properties(${TEST_OP} PROPERTIES LABELS "RUN_TYPE=DIST")
endforeach()
