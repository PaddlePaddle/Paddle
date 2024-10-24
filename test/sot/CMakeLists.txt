file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
set(SOT_ENVS SOT_LOG_LEVEL=0 COST_MODEL=False MIN_GRAPH_SIZE=0 STRICT_MODE=True
             FLAGS_cudnn_deterministic=True)

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP} ENVS ${SOT_ENVS})
endforeach()

if(WIN32)
  set_tests_properties(test_sot_resnet50_backward PROPERTIES TIMEOUT 420)
  set_tests_properties(test_sot_resnet PROPERTIES TIMEOUT 200)
endif()
