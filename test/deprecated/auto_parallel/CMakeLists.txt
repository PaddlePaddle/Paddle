# file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
# string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

if(WITH_DISTRIBUTE AND WITH_GPU)
  py_test_modules(test_pass_1F1B_deprecated MODULES test_pass_1F1B_deprecated)
  set_tests_properties(test_pass_1F1B_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)
  set_tests_properties(
    test_pass_1F1B_deprecated
    PROPERTIES ENVIRONMENT "FLAGS_new_executor_micro_batching=False")
endif()

set_pir_tests_properties()
