if(WITH_GPU)
  file(
    GLOB CINN_PIR_INFER_TEST
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "test_*.py")

  foreach(cinn_pir_test_name ${CINN_PIR_INFER_TEST})
    string(REGEX REPLACE ".py" "" cinn_pir_test_name ${cinn_pir_test_name})
    add_test(
      NAME ${cinn_pir_test_name}
      COMMAND
        ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
        FLAGS_prim_enable_dynamic=True FLAGS_prim_all=True
        FLAGS_check_infer_symbolic=1 FLAGS_enable_pir_api=1 ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/${cinn_pir_test_name}.py
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(${cinn_pir_test_name} PROPERTIES LABELS
                                                          "RUN_TYPE=CINN")
  endforeach()

  set_tests_properties(test_llama_inference PROPERTIES TIMEOUT 300)
  set_tests_properties(test_llama_forward PROPERTIES TIMEOUT 300)
  set_tests_properties(test_llama_postprocess PROPERTIES TIMEOUT 300)

endif()
