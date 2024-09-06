file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

list(REMOVE_ITEM TEST_OPS "test_fleet_with_asp_dynamic_deprecated")
list(REMOVE_ITEM TEST_OPS "test_fleet_with_asp_sharding_deprecated")

if(WITH_DISTRIBUTE)
  if(WITH_GPU OR WITH_XPU)
    py_test_modules(test_fleet_with_asp_dynamic_deprecated MODULES
                    test_fleet_with_asp_dynamic_deprecated ENVS ${dist_ENVS})
  endif()
endif()

if((WITH_DISTRIBUTE)
   AND (NOT WIN32)
   AND (NOT APPLE))
  if(WITH_GPU OR WITH_XPU)
    py_test_modules(test_fleet_with_asp_sharding_deprecated MODULES
                    test_fleet_with_asp_sharding_deprecated ENVS ${dist_ENVS})
  endif()
endif()

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach()

set_tests_properties(test_asp_pruning_dynamic_deprecated PROPERTIES TIMEOUT 30)
set_tests_properties(test_asp_pruning_static_deprecated PROPERTIES TIMEOUT 30)
set_tests_properties(test_asp_optimize_dynamic_deprecated PROPERTIES TIMEOUT 30)
set_tests_properties(test_asp_optimize_static_deprecated PROPERTIES TIMEOUT 30)
