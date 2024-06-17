set(TEST_PRIM_TRANS_PIR_CASES
    test_custom_vjp_trait test_decomp_op test_decompose_op test_vjp_prim
    test_batch_norm_shape_check test_builtin_slice)

foreach(target ${TEST_PRIM_TRANS_PIR_CASES})
  py_test_modules(${target} MODULES ${target} ENVS GLOG_v=1
                  FLAGS_enable_pir_in_executor=true)
endforeach()
