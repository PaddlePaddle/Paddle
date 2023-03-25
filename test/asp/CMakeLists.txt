file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

list(REMOVE_ITEM TEST_OPS "test_fleet_with_asp_static")
list(REMOVE_ITEM TEST_OPS "test_fleet_with_asp_dynamic")
list(REMOVE_ITEM TEST_OPS "test_fleet_with_asp_sharding")

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

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach()

if(WITH_DISTRIBUTE)
  if(WITH_GPU
     OR WITH_XPU
     OR WITH_ASCEND
     OR WITH_ASCEND_CL)
    py_test_modules(test_fleet_with_asp_dynamic MODULES
                    test_fleet_with_asp_dynamic ENVS ${dist_ENVS})
    py_test_modules(test_fleet_with_asp_static MODULES
                    test_fleet_with_asp_static ENVS ${dist_ENVS})
  endif()
endif()

if((WITH_DISTRIBUTE)
   AND (NOT WIN32)
   AND (NOT APPLE))
  if(WITH_GPU
     OR WITH_XPU
     OR WITH_ASCEND
     OR WITH_ASCEND_CL)
    py_test_modules(test_fleet_with_asp_sharding MODULES
                    test_fleet_with_asp_sharding ENVS ${dist_ENVS})
  endif()
endif()

set_tests_properties(test_asp_pruning_dynamic PROPERTIES TIMEOUT 30)
set_tests_properties(test_asp_pruning_static PROPERTIES TIMEOUT 30)
set_tests_properties(test_asp_optimize_dynamic PROPERTIES TIMEOUT 30)
set_tests_properties(test_asp_optimize_static PROPERTIES TIMEOUT 30)
