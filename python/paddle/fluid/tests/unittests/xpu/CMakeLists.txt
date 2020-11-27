file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

list(REMOVE_ITEM TEST_OPS test_concat_op_xpu)
list(REMOVE_ITEM TEST_OPS test_mean_op_xpu)

foreach(TEST_OP ${TEST_OPS})
    py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach(TEST_OP)

set_tests_properties(test_mul_op_xpu PROPERTIES TIMEOUT 120)
set_tests_properties(test_conv2d_op_xpu PROPERTIES TIMEOUT 120)
