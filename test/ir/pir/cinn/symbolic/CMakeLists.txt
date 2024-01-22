if(WITH_GPU)
  file(
    GLOB CINN_PIR_SYMBOLIC_TEST
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "test_*.py")

  list(REMOVE_ITEM CINN_PIR_SYMBOLIC_TEST test_cinn_reduce_symbolic_demo.py)

  foreach(cinn_pir_test_name ${CINN_PIR_SYMBOLIC_TEST})
    string(REGEX REPLACE ".py" "" cinn_pir_test_name ${cinn_pir_test_name})
    add_test(
      NAME ${cinn_pir_test_name}
      COMMAND
        ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
        FLAGS_enable_pir_api=1 FLAGS_cinn_bucket_compile=True
        ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/${cinn_pir_test_name}.py
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(${cinn_pir_test_name} PROPERTIES LABELS
                                                          "RUN_TYPE=CINN")
  endforeach()

  add_test(
    NAME test_cinn_reduce_symbolic_demo
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
      FLAGS_pir_apply_shape_optimization_pass=1 FLAGS_cinn_bucket_compile=True
      FLAGS_enable_pir_api=1 ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_cinn_reduce_symbolic_demo.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(test_cinn_reduce_symbolic_demo
                       PROPERTIES LABELS "RUN_TYPE=CINN")

endif()
