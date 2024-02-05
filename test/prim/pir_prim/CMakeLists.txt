set(TEST_PRIM_PURE_PIR_CASES
    test_prim_program
    test_prim_simpnet
    test_prim_custom_vjp
    test_prim_jit
    test_pir_prim_flags
    test_sink_decomp
    test_prim_skip_dynamic
    test_prim_dynamic
    test_prim_sub_graph_dynamic_shape
    test_decompose_control_flow)

foreach(target ${TEST_PRIM_PURE_PIR_CASES})
  py_test_modules(
    ${target}
    MODULES
    ${target}
    ENVS
    GLOG_v=1
    FLAGS_enable_pir_api=true
    FLAGS_prim_enable_dynamic=true)
endforeach()

set(TEST_PRIM_PURE_PIR_DY_SHAPE_CASES test_prim_jit_dynamic)

foreach(target ${TEST_PRIM_PURE_PIR_DY_SHAPE_CASES})
  py_test_modules(
    ${target}
    MODULES
    ${target}
    ENVS
    GLOG_v=1
    FLAGS_enable_pir_api=true
    FLAGS_prim_enable_dynamic=true
    FLAGS_cinn_bucket_compile=True
    FLAGS_pir_apply_shape_optimization_pass=1)
endforeach()

set_tests_properties(test_prim_jit_dynamic PROPERTIES LABELS "RUN_TYPE=CINN")

set(TEST_PRIM_TRANS_PIR_CASES test_custom_vjp_trait test_decomp_op
                              test_decompose_op test_vjp_prim)

foreach(target ${TEST_PRIM_TRANS_PIR_CASES})
  py_test_modules(${target} MODULES ${target} ENVS GLOG_v=1
                  FLAGS_enable_pir_in_executor=true)
endforeach()
