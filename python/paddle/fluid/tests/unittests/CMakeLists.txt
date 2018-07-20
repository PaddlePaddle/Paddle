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
    list(REMOVE_ITEM TEST_OPS test_dist_transpiler)
    list(REMOVE_ITEM TEST_OPS test_simple_dist_transpiler)
    list(REMOVE_ITEM TEST_OPS test_listen_and_serv_op)
    LIST(REMOVE_ITEM TEST_OPS test_dist_mnist)
    LIST(REMOVE_ITEM TEST_OPS test_dist_word2vec)
endif(NOT WITH_DISTRIBUTE)

if(WITH_DISTRIBUTE)
  add_subdirectory(distribute)
endif(WITH_DISTRIBUTE)

list(REMOVE_ITEM TEST_OPS test_seq_concat_op) # FIXME(helin): https://github.com/PaddlePaddle/Paddle/issues/8290
list(REMOVE_ITEM TEST_OPS test_modified_huber_loss_op) # FIXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/5184
list(REMOVE_ITEM TEST_OPS test_lstm_unit_op) # # FIXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/5185
list(REMOVE_ITEM TEST_OPS test_nce) # FIXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/7778
list(REMOVE_ITEM TEST_OPS test_recurrent_op) # FIXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/6152
list(REMOVE_ITEM TEST_OPS test_cond_op) # FIXME(qijun): https://github.com/PaddlePaddle/Paddle/issues/5101#issuecomment-339814957

list(REMOVE_ITEM TEST_OPS op_test) # op_test is a helper python file, not a test
list(REMOVE_ITEM TEST_OPS decorators) # decorators is a helper python file, not a test


list(REMOVE_ITEM TEST_OPS test_warpctc_op)
list(REMOVE_ITEM TEST_OPS test_dist_train)
list(REMOVE_ITEM TEST_OPS test_parallel_executor_crf)
list(REMOVE_ITEM TEST_OPS test_parallel_executor_fetch_feed)
foreach(TEST_OP ${TEST_OPS})
    py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach(TEST_OP)
py_test_modules(test_warpctc_op MODULES test_warpctc_op ENVS FLAGS_warpctc_dir=${WARPCTC_LIB_DIR} SERIAL)
if(WITH_DISTRIBUTE)
    py_test_modules(test_dist_train MODULES test_dist_train SERIAL)
    set_tests_properties(test_listen_and_serv_op PROPERTIES TIMEOUT 20)
    set_tests_properties(test_dist_mnist PROPERTIES TIMEOUT 180)
    set_tests_properties(test_dist_word2vec PROPERTIES TIMEOUT 180)
endif()
py_test_modules(test_parallel_executor_crf MODULES test_parallel_executor_crf SERIAL)
py_test_modules(test_parallel_executor_fetch_feed MODULES test_parallel_executor_fetch_feed SERIAL)
