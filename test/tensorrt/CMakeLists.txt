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
  set_tests_properties(test_converter_conv2d PROPERTIES TIMEOUT "100")
  set_tests_properties(test_converter_add PROPERTIES TIMEOUT "100")
  set_tests_properties(test_converter_batch_norm PROPERTIES TIMEOUT "100")
  set_tests_properties(test_converter_math PROPERTIES TIMEOUT "100")
  set_tests_properties(test_converter_ops PROPERTIES TIMEOUT "100")
  set_tests_properties(test_converter_activation PROPERTIES TIMEOUT "100")
  set_tests_properties(test_converter_others PROPERTIES TIMEOUT "100")
endif()
