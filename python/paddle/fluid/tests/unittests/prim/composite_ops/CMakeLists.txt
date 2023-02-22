file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")

file(
  GLOB TEST_OPS_GRAD
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*_grad.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
string(REPLACE ".py" "" TEST_OPS_GRAD "${TEST_OPS_GRAD}")

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP} ENVS ${GC_ENVS})
endforeach()

set_tests_properties(test_composite_batch_norm PROPERTIES TIMEOUT 120)
if(LINUX)
  set_tests_properties(test_composite_batch_norm_grad PROPERTIES TIMEOUT 120)
endif()
