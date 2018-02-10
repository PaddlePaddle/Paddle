file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

if(NOT WITH_DISTRIBUTE)
    list(REMOVE_ITEM TEST_OPS test_recv_op)
endif(NOT WITH_DISTRIBUTE)

list(REMOVE_ITEM TEST_OPS test_seq_concat_op) # FIXME(helin): https://github.com/PaddlePaddle/Paddle/issues/8290
list(REMOVE_ITEM TEST_OPS test_modified_huber_loss_op) # FIXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/5184 
list(REMOVE_ITEM TEST_OPS test_lstm_unit_op) # # FIXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/5185
list(REMOVE_ITEM TEST_OPS test_nce) # IXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/7778
list(REMOVE_ITEM TEST_OPS test_recurrent_op) # FIXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/6152
list(REMOVE_ITEM TEST_OPS test_cond_op) # FIXME(qijun): https://github.com/PaddlePaddle/Paddle/issues/5101#issuecomment-339814957
list(REMOVE_ITEM TEST_OPS test_detection_output_op) # FIXME: detection_output_op will be rewritten. This unittest should be

list(REMOVE_ITEM TEST_OPS op_test) # op_test is a helper python file, not a test
list(REMOVE_ITEM TEST_OPS decorators) # decorators is a helper python file, not a test

# test time consuming OPs in a separate process for expliot parallism
list(REMOVE_ITEM TEST_OPS test_warpctc_op)
py_test_modules(test_warpctc_op MODULES test_warpctc_op ENVS FLAGS_warpctc_dir=${WARPCTC_LIB_DIR})

list(REMOVE_ITEM TEST_OPS test_dyn_rnn)
py_test_modules(test_train_dyn_rnn MODULES test_dyn_rnn)

list(REMOVE_ITEM TEST_OPS test_mul_op)
py_test_modules(test_mul_op MODULES test_mul_op)

py_test_modules("test_all_ops" MODULES ${TEST_OPS})
