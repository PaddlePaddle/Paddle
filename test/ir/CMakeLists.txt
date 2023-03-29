file(
  GLOB TEST_IR_PASSES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_IR_PASSES "${TEST_IR_PASSES}")

if(((NOT WITH_GPU) AND (NOT WITH_ROCM))
   OR WIN32
   OR APPLE)
  list(REMOVE_ITEM TEST_IR_PASSES test_ir_fusion_group_pass)
endif()

function(py_test_modules TARGET_NAME)
  if(WITH_TESTING)
    set(options SERIAL)
    set(oneValueArgs "")
    set(multiValueArgs MODULES DEPS ENVS)
    cmake_parse_arguments(py_test_modules "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

    if(WITH_COVERAGE AND NOT (WITH_INCREMENTAL_COVERAGE
                              AND "$ENV{PADDLE_GIT_DIFF_PY_FILE}" STREQUAL ""))
      if(WITH_ASCEND_CL)
        add_test(
          NAME ${TARGET_NAME}
          COMMAND
            ${CMAKE_COMMAND} -E env
            PYTHONPATH=${PADDLE_BINARY_DIR}/python:$ENV{PYTHONPATH}
            ${py_test_modules_ENVS}
            COVERAGE_FILE=${PADDLE_BINARY_DIR}/python-coverage.data
            ${PYTHON_EXECUTABLE} -m coverage run --branch -p
            ${PADDLE_SOURCE_DIR}/tools/test_runner.py ${py_test_modules_MODULES}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
      else()
        add_test(
          NAME ${TARGET_NAME}
          COMMAND
            ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python
            ${py_test_modules_ENVS}
            COVERAGE_FILE=${PADDLE_BINARY_DIR}/python-coverage.data
            ${PYTHON_EXECUTABLE} -m coverage run --branch -p
            ${PADDLE_SOURCE_DIR}/tools/test_runner.py ${py_test_modules_MODULES}
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
            ${PADDLE_SOURCE_DIR}/tools/test_runner.py ${py_test_modules_MODULES}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
      else()
        add_test(
          NAME ${TARGET_NAME}
          COMMAND
            ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python
            ${py_test_modules_ENVS} ${PYTHON_EXECUTABLE}
            ${PADDLE_SOURCE_DIR}/tools/test_runner.py ${py_test_modules_MODULES}
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

foreach(target ${TEST_IR_PASSES})
  py_test_modules(${target} MODULES ${target})
  set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=INFER")
endforeach()

add_subdirectory(inference)

set_tests_properties(test_fuse_resnet_unit PROPERTIES TIMEOUT 120)
set_tests_properties(test_convert_to_mixed_precision PROPERTIES TIMEOUT 300)
