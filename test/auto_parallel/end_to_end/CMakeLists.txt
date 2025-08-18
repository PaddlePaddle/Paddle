# file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
# string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

if(WITH_DISTRIBUTE AND WITH_GPU)

  py_test_modules(test_e2e_co_shard MODULES test_e2e_co_shard)

endif()

set_pir_tests_properties()
