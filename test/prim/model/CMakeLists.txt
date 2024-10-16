file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

list(REMOVE_ITEM TEST_OP test_resnet_cinn test_resnet_prim_cinn
     test_prim_simplenet_cinn)

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP} ENVS ${GC_ENVS})
endforeach()

set_tests_properties(test_resnet_prim PROPERTIES TIMEOUT 850)
set_tests_properties(test_bert_prim PROPERTIES TIMEOUT 500)

if(WITH_CINN)
  set_tests_properties(test_bert_cinn PROPERTIES TIMEOUT 500)
  set_tests_properties(test_bert_prim_cinn PROPERTIES TIMEOUT 500)

  set_tests_properties(test_resnet_prim PROPERTIES LABELS "RUN_TYPE=CINN")
  set_tests_properties(test_bert_prim PROPERTIES LABELS "RUN_TYPE=CINN")
  set_tests_properties(test_bert_cinn PROPERTIES LABELS "RUN_TYPE=CINN")
  set_tests_properties(test_bert_prim_cinn PROPERTIES LABELS "RUN_TYPE=CINN")
endif()
