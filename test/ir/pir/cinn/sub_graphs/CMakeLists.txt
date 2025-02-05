if(WITH_GPU)
  set(PRIM_FORWARD_BLOCKLIST "pd_op.dropout")

  file(
    GLOB DYNAMIC_BUILD_TESTS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "test_*.py")
  string(REPLACE ".py" "" DYNAMIC_BUILD_TESTS "${DYNAMIC_BUILD_TESTS}")

  foreach(cinn_sub_graph_test_name ${DYNAMIC_BUILD_TESTS})
    add_test(
      NAME ${cinn_sub_graph_test_name}
      COMMAND
        ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
        FLAGS_prim_enable_dynamic=1 FLAGS_check_infer_symbolic=1
        FLAGS_prim_forward_blacklist=${PRIM_FORWARD_BLOCKLIST}
        FLAGS_cudnn_deterministic=true ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/${cinn_sub_graph_test_name}.py
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(${cinn_sub_graph_test_name} PROPERTIES LABELS
                                                                "RUN_TYPE=CINN")
    set_tests_properties(${cinn_sub_graph_test_name} PROPERTIES TIMEOUT 600)
  endforeach()

endif()
