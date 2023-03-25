if(WITH_IPU)
  function(py_test_modules TARGET_NAME)
    if(WITH_TESTING)
      set(options SERIAL)
      set(oneValueArgs "")
      set(multiValueArgs MODULES DEPS ENVS)
      cmake_parse_arguments(py_test_modules "${options}" "${oneValueArgs}"
                            "${multiValueArgs}" ${ARGN})

      if(WITH_COVERAGE
         AND NOT (WITH_INCREMENTAL_COVERAGE AND "$ENV{PADDLE_GIT_DIFF_PY_FILE}"
                                                STREQUAL ""))
        if(WITH_ASCEND_CL)
          add_test(
            NAME ${TARGET_NAME}
            COMMAND
              ${CMAKE_COMMAND} -E env
              PYTHONPATH=${PADDLE_BINARY_DIR}/python:$ENV{PYTHONPATH}
              ${py_test_modules_ENVS}
              COVERAGE_FILE=${PADDLE_BINARY_DIR}/python-coverage.data
              ${PYTHON_EXECUTABLE} -m coverage run --branch -p
              ${PADDLE_SOURCE_DIR}/tools/test_runner.py
              ${py_test_modules_MODULES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        else()
          add_test(
            NAME ${TARGET_NAME}
            COMMAND
              ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python
              ${py_test_modules_ENVS}
              COVERAGE_FILE=${PADDLE_BINARY_DIR}/python-coverage.data
              ${PYTHON_EXECUTABLE} -m coverage run --branch -p
              ${PADDLE_SOURCE_DIR}/tools/test_runner.py
              ${py_test_modules_MODULES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        endif()
      else()
        if(WITH_ASCEND_CL)
          add_test(
            NAME ${TARGET_NAME}
            COMMAND
              ${CMAKE_COMMAND} -E env
              PYTHONPATH=${PADDLE_BINARY_DIR}/python:$ENV{PYTHONPATH}
              ${py_test_modules_ENVS} ${PYTHON_EXECUTABLE}
              ${PADDLE_SOURCE_DIR}/tools/test_runner.py
              ${py_test_modules_MODULES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        else()
          add_test(
            NAME ${TARGET_NAME}
            COMMAND
              ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python
              ${py_test_modules_ENVS} ${PYTHON_EXECUTABLE}
              ${PADDLE_SOURCE_DIR}/tools/test_runner.py
              ${py_test_modules_MODULES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        endif()
      endif()

      if(py_test_modules_SERIAL)
        set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
      endif()
      if(WIN32)
        set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT 150)
      endif()
    endif()
  endfunction()

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
  set_tests_properties(test_elemetwise_x_op_ipu PROPERTIES TIMEOUT 600)
  set_tests_properties(test_reduce_x_op_ipu PROPERTIES TIMEOUT 800)
  set_tests_properties(test_save_load_ipu PROPERTIES TIMEOUT 600)
  set_tests_properties(test_activation_ops_ipu PROPERTIES TIMEOUT 600)
  set_tests_properties(test_unary_ops_ipu PROPERTIES TIMEOUT 600)
  set_tests_properties(test_greater_op_ipu PROPERTIES TIMEOUT 400)
  set_tests_properties(test_conv2d_transpose_op_ipu PROPERTIES TIMEOUT 300)
  set_tests_properties(test_model_parallel_ipu PROPERTIES TIMEOUT 300)

  add_subdirectory(custom_ops)

endif()
