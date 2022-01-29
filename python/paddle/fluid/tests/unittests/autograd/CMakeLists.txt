file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
set(GC_ENVS FLAGS_eager_delete_tensor_gb=0.0)

foreach(TEST_OP ${TEST_OPS})
    py_test_modules(${TEST_OP} MODULES ${TEST_OP} ENVS ${GC_ENVS})
endforeach(TEST_OP)

set_tests_properties(test_jacobian PROPERTIES TIMEOUT 50)
set_tests_properties(test_hessian PROPERTIES TIMEOUT 50)
set_tests_properties(test_vhp PROPERTIES TIMEOUT 50)
