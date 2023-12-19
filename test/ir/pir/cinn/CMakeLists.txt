add_subdirectory(adt)
add_subdirectory(symbolic)

if(WITH_GPU)
  file(
    GLOB CINN_PIR_TEST
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "test_*.py")
  string(REPLACE ".py" "" CINN_PIR_TEST "${CINN_PIR_TEST}")

  list(REMOVE_ITEM CINN_PIR_TEST test_subgraph_checker)

  foreach(cinn_pir_test_name ${CINN_PIR_TEST})
    add_test(
      NAME ${cinn_pir_test_name}
      COMMAND
        ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
        FLAGS_enable_pir_api=1 FLAGS_prim_all=True ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/${cinn_pir_test_name}.py
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(${cinn_pir_test_name} PROPERTIES LABELS
                                                          "RUN_TYPE=CINN")
  endforeach()

  add_test(
    NAME test_subgraph_checker
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
      FLAGS_enable_pir_api=1 ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_subgraph_checker.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(test_subgraph_checker PROPERTIES LABELS "RUN_TYPE=CINN")

endif()
