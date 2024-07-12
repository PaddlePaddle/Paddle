file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

function(py_test_modules TARGET_NAME)
  if(WITH_TESTING)
    set(options SERIAL)
    set(oneValueArgs "")
    set(multiValueArgs MODULES DEPS ENVS)
    cmake_parse_arguments(py_test_modules "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

    string(REGEX MATCH "_deprecated\.py$" DEPRECATED_MODULES
                 "${py_test_modules_MODULES}")
    string(REGEX MATCH "_deprecated$" DEPRECATED_TARGET_NAME "${TARGET_NAME}")
    set(FLAGS_PIR_MODE "")
    if((NOT "${DEPRECATED_MODULES}" STREQUAL "")
       OR (NOT "${DEPRECATED_TARGET_NAME}" STREQUAL ""))
      set(FLAGS_PIR_MODE FLAGS_enable_pir_api=0)
    endif()
    if(WITH_COVERAGE AND NOT (WITH_INCREMENTAL_COVERAGE
                              AND "$ENV{PADDLE_GIT_DIFF_PY_FILE}" STREQUAL ""))
      add_test(
        NAME ${TARGET_NAME}
        COMMAND
          ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python
          ${py_test_modules_ENVS} ${FLAGS_PIR_MODE}
          COVERAGE_FILE=${PADDLE_BINARY_DIR}/python-coverage.data
          ${PYTHON_EXECUTABLE} -m coverage run --branch -p
          ${PADDLE_SOURCE_DIR}/tools/test_runner.py ${py_test_modules_MODULES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    else()
      add_test(
        NAME ${TARGET_NAME}
        COMMAND
          ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python
          ${py_test_modules_ENVS} ${FLAGS_PIR_MODE} ${PYTHON_EXECUTABLE}
          ${PADDLE_SOURCE_DIR}/tools/test_runner.py ${py_test_modules_MODULES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    if(py_test_modules_SERIAL)
      set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
    endif()
    if(WIN32)
      set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT 150)
    endif()
  endif()
endfunction()

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach()

if(APPLE)
  set_tests_properties(test_model_cast_to_bf16 PROPERTIES TIMEOUT 300)
endif()
