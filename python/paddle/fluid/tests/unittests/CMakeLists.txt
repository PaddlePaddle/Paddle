file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

# The MKLDNN tests are skiped when the MKLDNN flag is OFF
if(NOT WITH_MKLDNN)
    foreach(src ${TEST_OPS})
        if(${src} MATCHES ".*_mkldnn_op$")
            list(REMOVE_ITEM TEST_OPS ${src})
        endif()
    endforeach()
endif(NOT WITH_MKLDNN)

if(NOT WITH_DISTRIBUTE)
    list(REMOVE_ITEM TEST_OPS test_recv_op)
endif(NOT WITH_DISTRIBUTE)

list(REMOVE_ITEM TEST_OPS test_seq_concat_op) # FIXME(helin): https://github.com/PaddlePaddle/Paddle/issues/8290
list(REMOVE_ITEM TEST_OPS test_modified_huber_loss_op) # FIXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/5184
list(REMOVE_ITEM TEST_OPS test_lstm_unit_op) # # FIXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/5185
list(REMOVE_ITEM TEST_OPS test_nce) # FIXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/7778
list(REMOVE_ITEM TEST_OPS test_recurrent_op) # FIXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/6152
list(REMOVE_ITEM TEST_OPS test_cond_op) # FIXME(qijun): https://github.com/PaddlePaddle/Paddle/issues/5101#issuecomment-339814957

list(REMOVE_ITEM TEST_OPS op_test) # op_test is a helper python file, not a test
list(REMOVE_ITEM TEST_OPS decorators) # decorators is a helper python file, not a test

function(py_test_modules TARGET_NAME)
  if(WITH_TESTING)
    set(options SERIAL)
    set(oneValueArgs "")
    set(multiValueArgs MODULES DEPS ENVS)
    cmake_parse_arguments(py_test_modules "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_test(NAME ${TARGET_NAME}
             COMMAND env PYTHONPATH=${PADDLE_BINARY_DIR}/python ${py_test_modules_ENVS}
             ${PYTHON_EXECUTABLE} ${PADDLE_SOURCE_DIR}/tools/test_runner.py ${py_test_modules_MODULES}
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    if (py_test_modules_SERIAL)
        set_property(TEST ${TARGET_NAME} PROPERTY SERIAL 1)
    endif()
  endif()
endfunction()
list(REMOVE_ITEM TEST_OPS test_warpctc_op)
list(REMOVE_ITEM TEST_OPS test_dist_train)
list(REMOVE_ITEM TEST_OPS test_parallel_executor_crf)
list(REMOVE_ITEM TEST_OPS test_parallel_executor_fetch_feed)
foreach(TEST_OP ${TEST_OPS})
    py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach(TEST_OP)
py_test_modules(test_warpctc_op MODULES test_warpctc_op ENVS FLAGS_warpctc_dir=${WARPCTC_LIB_DIR} SERIAL)
py_test_modules(test_dist_train MODULES test_dist_train SERIAL)
py_test_modules(test_parallel_executor_crf MODULES test_parallel_executor_crf SERIAL)
py_test_modules(test_parallel_executor_fetch_feed MODULES test_parallel_executor_fetch_feed SERIAL)
set_tests_properties(test_listen_and_serv_op PROPERTIES TIMEOUT 20)
