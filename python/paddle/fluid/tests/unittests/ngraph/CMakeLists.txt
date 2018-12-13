file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

# The nGraph tests are skiped when the nGraph flag is OFF
if(NOT WITH_NGRAPH)
    foreach(src ${TEST_OPS})
        if(${src} MATCHES ".*_ngraph_op$")
            list(REMOVE_ITEM TEST_OPS ${src})
        endif()
    endforeach()
endif(NOT WITH_NGRAPH)

function(py_test_modules TARGET_NAME)
  if(WITH_TESTING)
    set(options SERIAL)
    set(oneValueArgs "")
    set(multiValueArgs MODULES DEPS ENVS)
    cmake_parse_arguments(py_test_modules "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set(use_ngraph "true")
    add_test(NAME ${TARGET_NAME}
        COMMAND ${CMAKE_COMMAND} -E env FLAGS_use_ngraph=${use_ngraph} PYTHONPATH=${PADDLE_BINARY_DIR}/python ${py_test_modules_ENVS}
        ${PYTHON_EXECUTABLE} ${PADDLE_SOURCE_DIR}/tools/test_runner.py ${py_test_modules_MODULES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    if (py_test_modules_SERIAL)
        set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
    endif()
    set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT 600)
  endif()
endfunction()

foreach(TEST_OP ${TEST_OPS})
    py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach(TEST_OP)
