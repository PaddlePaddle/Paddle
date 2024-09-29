remove_definitions(-DPADDLE_DLL_EXPORT)
set(CC_TESTS_DIR
    ${PADDLE_BINARY_DIR}/test/cpp
    CACHE INTERNAL "c++ tests directory")
set(PYTHON_TESTS_DIR
    ${PADDLE_BINARY_DIR}/test
    CACHE INTERNAL "python tests directory")

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

function(bash_test_modules TARGET_NAME)
  if(NOT WITH_TESTING)
    return()
  endif()

  set(options SERIAL)
  set(oneValueArgs TIMEOUT START_BASH)
  set(multiValueArgs DEPS ENVS LABELS)
  cmake_parse_arguments(bash_test_modules "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  set(timeout 350)
  if(${bash_test_modules_TIMEOUT})
    set(timeout ${bash_test_modules_TIMEOUT})
  endif()

  if(WITH_COVERAGE)
    add_test(
      NAME ${TARGET_NAME}
      COMMAND
        ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python
        TEST_TARGET_NAME=${TARGET_NAME} TEST_TIMEOUT=${timeout}
        ${bash_test_modules_ENVS} WITH_COVERAGE=ON
        COVERAGE_FILE=${PADDLE_BINARY_DIR}/python-coverage.data bash
        ${CMAKE_CURRENT_BINARY_DIR}/${bash_test_modules_START_BASH}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  else()
    add_test(
      NAME ${TARGET_NAME}
      COMMAND
        ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python
        TEST_TARGET_NAME=${TARGET_NAME} TEST_TIMEOUT=${timeout}
        ${bash_test_modules_ENVS} bash
        ${CMAKE_CURRENT_BINARY_DIR}/${bash_test_modules_START_BASH}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()

  if(bash_test_modules_SERIAL)
    set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
  endif()

  if(bash_test_modules_LABELS)
    set_tests_properties(${TARGET_NAME} PROPERTIES LABELS
                                                   ${bash_test_modules_LABELS})
  endif()
endfunction()

function(set_pir_tests_properties)
  file(STRINGS "${CMAKE_SOURCE_DIR}/test/white_list/pir_op_test_white_list"
       PIR_OP_TESTS)
  foreach(IR_OP_TEST ${PIR_OP_TESTS})
    if(TEST ${IR_OP_TEST})
      set_property(
        TEST ${IR_OP_TEST}
        APPEND
        PROPERTY ENVIRONMENT "FLAGS_PIR_OPTEST_WHITE_LIST=True")
    endif()
  endforeach()

  file(STRINGS "${CMAKE_SOURCE_DIR}/test/white_list/pir_op_test_no_check_list"
       PIR_OP_NO_CHECK_TESTS)
  foreach(IR_OP_TEST ${PIR_OP_NO_CHECK_TESTS})
    if(TEST ${IR_OP_TEST})
      set_property(
        TEST ${IR_OP_TEST}
        APPEND
        PROPERTY ENVIRONMENT "FLAGS_PIR_NO_CHECK=True")
    endif()
  endforeach()

  file(STRINGS
       "${CMAKE_SOURCE_DIR}/test/white_list/pir_op_test_precision_white_list"
       PIR_OP_RELAXED_TESTS)
  foreach(IR_OP_TEST ${PIR_OP_RELAXED_TESTS})
    if(TEST ${IR_OP_TEST})
      set_property(
        TEST ${IR_OP_TEST}
        APPEND
        PROPERTY ENVIRONMENT "FLAGS_PIR_OPTEST_RELAX_CHECK=True")
    endif()
  endforeach()

endfunction()

if(WITH_TESTING)
  if(WIN32 AND WIN_UNITTEST_LEVEL LESS 2)
    message(STATUS "Skip tests unrelated to CUDA/TRT")
  else()
    add_subdirectory(asp)
    add_subdirectory(custom_op)
    add_subdirectory(prim)
    add_subdirectory(standalone_executor)
    add_subdirectory(tokenizer)
  endif()

  add_subdirectory(book)
  add_subdirectory(contrib)
  add_subdirectory(cpp)
  add_subdirectory(ir)
  add_subdirectory(legacy_test)
  add_subdirectory(quantization)
  add_subdirectory(rnn)
  add_subdirectory(sequence)

  if(WITH_DISTRIBUTE)
    add_subdirectory(auto_parallel)
    add_subdirectory(collective)
  endif()
  if(WITH_ONEDNN)
    add_subdirectory(mkldnn)
  endif()

endif()

set_pir_tests_properties()
