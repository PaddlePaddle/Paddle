file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP} ENVS ${GC_ENVS})
endforeach()

set_tests_properties(test_resnet_prim_cinn PROPERTIES TIMEOUT 850)
set_tests_properties(test_bert_prim_cinn PROPERTIES TIMEOUT 500)
set_tests_properties(test_prim_simplenet_cinn PROPERTIES TIMEOUT 120)

if(WITH_CINN)
  set_tests_properties(test_resnet_prim_cinn PROPERTIES LABELS "RUN_TYPE=CINN")
  set_tests_properties(test_bert_prim_cinn PROPERTIES LABELS "RUN_TYPE=CINN")
  set_tests_properties(test_prim_simplenet_cinn PROPERTIES LABELS
                                                           "RUN_TYPE=CINN")
endif()
