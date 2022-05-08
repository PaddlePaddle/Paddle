if(WITH_IPU)
    file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
    string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

    foreach(TEST_OP ${TEST_OPS})
        py_test_modules(${TEST_OP} MODULES ${TEST_OP})
        # set all UTs timeout to 200s
        set_tests_properties(${TEST_OP} PROPERTIES TIMEOUT 200)    
    endforeach(TEST_OP)

    set_tests_properties(test_conv_op_ipu PROPERTIES TIMEOUT 300)
    set_tests_properties(test_elemetwise_x_op_ipu PROPERTIES TIMEOUT 300)
    set_tests_properties(test_reduce_x_op_ipu PROPERTIES TIMEOUT 600)
endif()
