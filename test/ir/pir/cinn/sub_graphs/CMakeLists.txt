if(WITH_GPU)
  file(
    GLOB CINN_SUB_GRAPH_TEST
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "test_*.py")

  foreach(cinn_sub_graph_test_name ${CINN_SUB_GRAPH_TEST})
    string(REGEX REPLACE ".py" "" cinn_sub_graph_test_name
                         ${cinn_sub_graph_test_name})
    add_test(
      NAME ${cinn_sub_graph_test_name}
      COMMAND
        ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
        FLAGS_enable_pir_api=1 ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/${cinn_sub_graph_test_name}.py
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(${cinn_sub_graph_test_name} PROPERTIES LABELS
                                                                "RUN_TYPE=CINN")
  endforeach()

endif()
