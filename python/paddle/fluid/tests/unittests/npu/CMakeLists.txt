file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

if (WITH_ASCEND_CL)
    foreach(TEST_OP ${TEST_OPS})
        py_test_modules(${TEST_OP} MODULES ${TEST_OP})
    endforeach(TEST_OP)

    # NOTE: NPU `get_float_status` read the value from register, During the test,
    # it is found that this register will be overwritten by any program on the card.
    # In order to prevent the interference of nan/inf in the other unittests, we
    # need to set the unittests related to `float_status` to exclusive.
    set_tests_properties(test_amp_check_finite_and_scale_op_npu PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
    set_tests_properties(test_flags_check_nan_inf_npu PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
    set_tests_properties(test_float_status_op_npu PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")

    # Note: the following test cases has running time more than 120s
    set_tests_properties(test_nearest_interp_op_npu PROPERTIES TIMEOUT 200)
    set_tests_properties(test_nearest_interp_v2_op_npu PROPERTIES TIMEOUT 200)
    set_tests_properties(test_bilinear_interp_v2_op_npu PROPERTIES TIMEOUT 200)
    set_tests_properties(test_stack_op_npu PROPERTIES TIMEOUT 300)
    set_tests_properties(test_conv2d_transpose_op_npu PROPERTIES TIMEOUT 200)
    set_tests_properties(test_conv2d_op_npu PROPERTIES TIMEOUT 300)
    set_tests_properties(test_matmulv2_op_npu PROPERTIES TIMEOUT 300)
    set_tests_properties(test_elementwise_add_op_npu PROPERTIES TIMEOUT 200)
endif()
