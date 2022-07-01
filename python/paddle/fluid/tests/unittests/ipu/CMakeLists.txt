if(WITH_IPU)
  file(
    GLOB TEST_OPS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "test_*.py")
  string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

  foreach(TEST_OP ${TEST_OPS})
    py_test_modules(${TEST_OP} MODULES ${TEST_OP})
    # set all UTs timeout to 200s
    set_tests_properties(${TEST_OP} PROPERTIES TIMEOUT 200)
  endforeach()

  set_tests_properties(test_conv_op_ipu PROPERTIES TIMEOUT 300)
  set_tests_properties(test_elemetwise_x_op_ipu PROPERTIES TIMEOUT 300)
  set_tests_properties(test_reduce_x_op_ipu PROPERTIES TIMEOUT 600)
  set_tests_properties(test_save_load_ipu PROPERTIES TIMEOUT 600)

  file(GLOB INFERENCE_TESTS "inference/test_*.py")
  string(REPLACE ".py" "" INFERENCE_TESTS "${INFERENCE_TESTS}")
  foreach(INFERENCE_TEST ${INFERENCE_TESTS})
    py_test_modules(${INFERENCE_TEST} MODULES ${INFERENCE_TEST})
    set_tests_properties(${INFERENCE_TEST} PROPERTIES TIMEOUT 600)
  endforeach()

  file(GLOB CUSTOM_OP_TESTS "custom_op/test_*.py")
  list(
    APPEND CUSTOM_OP_TESTS
    "${CMAKE_CURRENT_SOURCE_DIR}/custom_op/deprecated/test_custom_nllloss.py")
  string(REPLACE ".py" "" CUSTOM_OP_TESTS "${CUSTOM_OP_TESTS}")
  foreach(CUSTOM_OP_TEST ${CUSTOM_OP_TESTS})
    py_test_modules(${CUSTOM_OP_TEST} MODULES ${CUSTOM_OP_TEST})
  endforeach()

endif()
