set(TEST_PRIM_PURE_PIR_CASES
    test_prim_program
    test_prim_simpnet
    test_prim_custom_vjp
    test_prim_jit
    test_pir_prim_flags
    test_sink_decomp
    test_prim_dynamic)

foreach(target ${TEST_PRIM_PURE_PIR_CASES})
  py_test_modules(${target} MODULES ${target} ENVS GLOG_v=1
                  FLAGS_enable_pir_api=true)
endforeach()

file(
  GLOB TEST_PRIM_TRANS_PIR_CASES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_PRIM_TRANS_PIR_CASES
               "${TEST_PRIM_TRANS_PIR_CASES}")

list(REMOVE_ITEM TEST_PRIM_TRANS_PIR_CASES ${TEST_PRIM_PURE_PIR_CASES})

foreach(target ${TEST_PRIM_TRANS_PIR_CASES})
  py_test_modules(${target} MODULES ${target} ENVS GLOG_v=1
                  FLAGS_enable_pir_in_executor=true)
endforeach()
