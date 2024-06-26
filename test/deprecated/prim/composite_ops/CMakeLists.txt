file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")

string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP} ENVS ${GC_ENVS})
endforeach()

set_tests_properties(test_composite_batch_norm_deprecated PROPERTIES TIMEOUT
                                                                     120)
set_tests_properties(test_composite_mean_grad_deprecated PROPERTIES TIMEOUT 120)
if(LINUX)
  set_tests_properties(test_composite_batch_norm_grad_deprecated
                       PROPERTIES TIMEOUT 120)
endif()
