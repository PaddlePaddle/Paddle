file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
set(GC_ENVS FLAGS_eager_delete_tensor_gb=0.0)

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP} ENVS ${GC_ENVS})
endforeach()

set_tests_properties(test_comp_sum_grad_deprecated PROPERTIES TIMEOUT 60)
set_tests_properties(test_comp_tanh_grad PROPERTIES TIMEOUT 60)
set_tests_properties(test_comp_div_grad_deprecated PROPERTIES TIMEOUT 60)
set_tests_properties(test_comp_add_grad_deprecated PROPERTIES TIMEOUT 60)
set_tests_properties(test_comp_sub_grad_deprecated PROPERTIES TIMEOUT 60)
set_tests_properties(test_comp_add_tanh_grad_deprecated PROPERTIES TIMEOUT 60)
set_tests_properties(test_comp_sqrt_grad PROPERTIES TIMEOUT 60)
