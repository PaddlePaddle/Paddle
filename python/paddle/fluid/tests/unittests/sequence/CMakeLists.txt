file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

if (NOT ${WITH_GPU})
    LIST(REMOVE_ITEM TEST_OPS test_sequence_pool_all) # TODO(shenliang03): test_sequence_pool_all support CPU device in future
endif()

foreach(TEST_OP ${TEST_OPS})
    py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach(TEST_OP)
