if(NOT WIN32 AND TENSORRT_FOUND)
  file(
    GLOB TEST_OPS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "test_*.py")
  string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

  foreach(TEST_OP ${TEST_OPS})
    py_test_modules(${TEST_OP} MODULES ${TEST_OP})
  endforeach()
  set_tests_properties(test_converter_model_bert PROPERTIES TIMEOUT "500")
  set_tests_properties(test_converter_model_dummy PROPERTIES TIMEOUT "500")
  set_tests_properties(test_converter_model_resnet50 PROPERTIES TIMEOUT "500")
endif()
