if(WITH_GPU)
  file(
    GLOB CINN_PIR_TEST
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "test_*.py")

  foreach(cinn_pir_test_name ${CINN_PIR_TEST})
    string(REGEX REPLACE ".py" "" cinn_pir_test_name ${cinn_pir_test_name})
    string(FIND "${cinn_pir_test_name}" "symbolic" index_of_substr)

    message(${index_of_substr} "liuchao liuchao" ${cinn_pir_test_name})
    if(index_of_substr EQUAL -1)
      add_test(
        NAME ${cinn_pir_test_name}
        COMMAND
          ${CMAKE_COMMAND} -E env
          PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
          FLAGS_enable_pir_api=1 FLAGS_prim_all=True ${PYTHON_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/${cinn_pir_test_name}.py
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    else()
      add_test(
        NAME ${cinn_pir_test_name}
        COMMAND
          ${CMAKE_COMMAND} -E env
          PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
          FLAGS_cinn_bucket_compile=True FLAGS_enable_pir_api=1
          FLAGS_prim_all=True ${PYTHON_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/${cinn_pir_test_name}.py
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    endif()
    set_tests_properties(${cinn_pir_test_name} PROPERTIES LABELS
                                                          "RUN_TYPE=CINN")
  endforeach()

endif()
