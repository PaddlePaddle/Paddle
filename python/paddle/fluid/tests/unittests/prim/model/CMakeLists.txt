file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP} ENVS ${GC_ENVS})
endforeach()

set_tests_properties(test_resnet_prim_cinn PROPERTIES TIMEOUT 400)
set_tests_properties(test_bert_prim_cinn PROPERTIES TIMEOUT 400)

if(WITH_CINN)
  set_tests_properties(test_resnet_prim_cinn PROPERTIES LABELS "RUN_TYPE=CINN")
  set_tests_properties(
    test_bert_prim_cinn
    PROPERTIES LABELS "RUN_TYPE=CINN" ENVIRONMENT
               FLAGS_deny_cinn_ops="dropout;layer_norm;slice")
endif()

if(WITH_TESTING)
  add_custom_target(
    download_bert_training_data ALL
    COMMAND wget -nc -q --no-check-certificate
            https://paddle-ci.gz.bcebos.com/prim_cinn/bert_training_data.npz
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif()
