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
        FLAGS_cinn_new_group_scheduler=1 FLAGS_enable_pir_api=1
        FLAGS_cinn_bucket_compile=1 FLAGS_group_schedule_tiling_first=1
        FLAGS_cudnn_deterministic=true ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/${cinn_sub_graph_test_name}.py
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(${cinn_sub_graph_test_name} PROPERTIES LABELS
                                                                "RUN_TYPE=CINN")
  endforeach()
  set_tests_properties(test_sub_graph_3 PROPERTIES TIMEOUT 300)
  set_tests_properties(test_sub_graph_54 PROPERTIES TIMEOUT 300)
  set_tests_properties(test_sub_graph_30 PROPERTIES TIMEOUT 300)

endif()
