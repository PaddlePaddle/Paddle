file(
  GLOB TEST_INTERP_CASES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_INTERP_CASES "${TEST_INTERP_CASES}")

set(TEST_IR_SYSTEM_CASES
    test_build_model test_pd_inplace_pass test_symbol_overload
    test_new_ir_to_static test_stop_gradient test_override_operator)
list(REMOVE_ITEM TEST_INTERP_CASES ${TEST_IR_SYSTEM_CASES})

foreach(target ${TEST_INTERP_CASES})
  py_test_modules(${target} MODULES ${target} ENVS GLOG_v=1
                  FLAGS_enable_new_ir_in_executor=true)
endforeach()

foreach(target ${TEST_IR_SYSTEM_CASES})
  py_test_modules(${target} MODULES ${target} ENVS FLAGS_enable_pir_api=true)
endforeach()

set_tests_properties(test_pd_inplace_pass PROPERTIES TIMEOUT 60)
