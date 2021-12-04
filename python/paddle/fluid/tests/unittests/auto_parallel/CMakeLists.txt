file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

if(NOT WITH_NCCL)
    list(REMOVE_ITEM TEST_OPS test_auto_parallel_launch)
    list(APPEND DIST_TEST_OPS test_auto_parallel_launch)
endif()
