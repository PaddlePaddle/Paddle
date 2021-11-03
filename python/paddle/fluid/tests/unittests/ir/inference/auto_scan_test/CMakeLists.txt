if (WITH_MKLDNN AND WITH_GPU)

    # pass test
    file(GLOB TEST_PASSES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*_pass.py")
    string(REPLACE ".py" "" TEST_PASSES "${TEST_PASSES}")
    foreach(target ${TEST_PASSES})
        py_test_modules(${target} MODULES ${target})
        # set_tests_properties(${target} PROPERTIES TIMEOUT 300)
    endforeach()

    # mkldnn op test
    file(GLOB TEST_MKLDNN_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_mkldnn_*_op.py")
    string(REPLACE ".py" "" TEST_MKLDNN_OPS "${TEST_MKLDNN_OPS}")
    foreach(target ${TEST_MKLDNN_OPS})
        py_test_modules(${target} MODULES ${target})
        # set_tests_properties(${target} PROPERTIES TIMEOUT 300)
    endforeach()

endif()
