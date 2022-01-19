file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

list(REMOVE_ITEM TEST_OPS "test_fleet_with_asp")
list(REMOVE_ITEM TEST_OPS "test_fleet_with_asp_amp")
list(REMOVE_ITEM TEST_OPS "test_fleet_with_asp_sharding")

foreach(TEST_OP ${TEST_OPS})
    py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach(TEST_OP)

if(WITH_DISTRIBUTE)
    py_test_modules(test_fleet_with_asp MODULES test_fleet_with_asp ENVS ${dist_ENVS})
    py_test_modules(test_fleet_with_asp_amp MODULES test_fleet_with_asp_amp ENVS ${dist_ENVS})
endif()

if((WITH_DISTRIBUTE) AND (NOT WIN32) AND (NOT APPLE))
    py_test_modules(test_fleet_with_asp_sharding MODULES test_fleet_with_asp_sharding ENVS ${dist_ENVS})
endif()
