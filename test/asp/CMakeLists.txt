file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

list(REMOVE_ITEM TEST_OPS "test_fleet_with_asp_static")
list(REMOVE_ITEM TEST_OPS "test_fleet_with_asp_dynamic")
list(REMOVE_ITEM TEST_OPS "test_fleet_with_asp_sharding")

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach()

if(WITH_DISTRIBUTE)
  if(WITH_GPU OR WITH_XPU)
    py_test_modules(test_fleet_with_asp_dynamic MODULES
                    test_fleet_with_asp_dynamic ENVS ${dist_ENVS})
    py_test_modules(test_fleet_with_asp_static MODULES
                    test_fleet_with_asp_static ENVS ${dist_ENVS})
  endif()
endif()

if((WITH_DISTRIBUTE)
   AND (NOT WIN32)
   AND (NOT APPLE))
  if(WITH_GPU OR WITH_XPU)
    py_test_modules(test_fleet_with_asp_sharding MODULES
                    test_fleet_with_asp_sharding ENVS ${dist_ENVS})
  endif()
endif()
