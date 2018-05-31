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
list(REMOVE_ITEM TEST_OPS test_nce) # IXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/7778
list(REMOVE_ITEM TEST_OPS test_recurrent_op) # FIXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/6152
list(REMOVE_ITEM TEST_OPS test_cond_op) # FIXME(qijun): https://github.com/PaddlePaddle/Paddle/issues/5101#issuecomment-339814957

list(REMOVE_ITEM TEST_OPS op_test) # op_test is a helper python file, not a test
list(REMOVE_ITEM TEST_OPS decorators) # decorators is a helper python file, not a test

function(py_test_modules TARGET_NAME)
  if(WITH_TESTING)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs MODULES DEPS ARGS ENVS)
    cmake_parse_arguments(py_test_modules "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_test(NAME ${TARGET_NAME}
             COMMAND env PYTHONPATH=${PADDLE_BINARY_DIR}/python ${py_test_modules_ENVS}
             ${PYTHON_EXECUTABLE} -u -m unittest --verbose ${py_test_modules_MODULES} ${py_test_modules_ARGS}
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()
endfunction()

list(REMOVE_ITEM TEST_OPS test_sequence_expand)

# test time consuming OPs in a separate process for expliot parallism
list(REMOVE_ITEM TEST_OPS test_parallel_executor)
list(REMOVE_ITEM TEST_OPS test_warpctc_op)
list(REMOVE_ITEM TEST_OPS test_dyn_rnn)
list(REMOVE_ITEM TEST_OPS test_mul_op)

# tests that need to be run in separate process.
list(REMOVE_ITEM TEST_OPS test_multihead_attention)
list(REMOVE_ITEM TEST_OPS test_calc_gradient)
list(REMOVE_ITEM TEST_OPS test_while_op)
list(REMOVE_ITEM TEST_OPS test_lod_array_length_op)
list(REMOVE_ITEM TEST_OPS test_reorder_lod_tensor)
list(REMOVE_ITEM TEST_OPS test_profiler)
list(REMOVE_ITEM TEST_OPS test_nvprof)
list(REMOVE_ITEM TEST_OPS test_normalization_wrapper)
list(REMOVE_ITEM TEST_OPS test_executor_and_mul)
list(REMOVE_ITEM TEST_OPS test_assign_value_op)
list(REMOVE_ITEM TEST_OPS test_array_read_write_op)
list(REMOVE_ITEM TEST_OPS test_lod_rank_table)
list(REMOVE_ITEM TEST_OPS test_weight_normalization)
list(REMOVE_ITEM TEST_OPS test_conditional_block)
list(REMOVE_ITEM TEST_OPS test_parameter)
list(REMOVE_ITEM TEST_OPS test_registry)
list(REMOVE_ITEM TEST_OPS test_fetch_var)
list(REMOVE_ITEM TEST_OPS test_parallel_op)
list(REMOVE_ITEM TEST_OPS test_dynrnn_static_input)
list(REMOVE_ITEM TEST_OPS test_dist_train)

# tests that can be bundled together in one python process for speed.
if(WITH_FAST_BUNDLE_TEST)
    py_test_modules("test_all_ops" MODULES ${TEST_OPS})
else()
    foreach(TEST_OP ${TEST_OPS})
        py_test_modules(${TEST_OP} MODULES ${TEST_OP})
    endforeach(TEST_OP)
endif(WITH_FAST_BUNDLE_TEST)

#
py_test_modules(test_sequence_expand MODULES test_sequence_expand)
# tests with high overhead
py_test_modules(test_parallel_executor MODULES test_parallel_executor)
py_test_modules(test_warpctc_op MODULES test_warpctc_op ENVS FLAGS_warpctc_dir=${WARPCTC_LIB_DIR})
py_test_modules(test_train_dyn_rnn MODULES test_dyn_rnn)
py_test_modules(test_mul_op MODULES test_mul_op)

# tests that need to be run in separate process.
py_test_modules(test_multihead_attention MODULES test_multihead_attention)
py_test_modules(test_calc_gradient MODULES test_calc_gradient)
py_test_modules(test_while_op MODULES test_while_op)
py_test_modules(test_lod_array_length_op MODULES test_lod_array_length_op)
py_test_modules(test_reorder_lod_tensor MODULES test_reorder_lod_tensor)
py_test_modules(test_profiler MODULES test_profiler)
py_test_modules(test_nvprof MODULES test_nvprof)
py_test_modules(test_normalization_wrapper MODULES test_normalization_wrapper)
py_test_modules(test_executor_and_mul MODULES test_executor_and_mul)
py_test_modules(test_assign_value_op MODULES test_assign_value_op)
py_test_modules(test_array_read_write_op MODULES test_array_read_write_op)
py_test_modules(test_lod_rank_table MODULES test_lod_rank_table)
py_test_modules(test_weight_normalization MODULES test_weight_normalization)
py_test_modules(test_conditional_block MODULES test_conditional_block)
py_test_modules(test_parameter MODULES test_parameter)
py_test_modules(test_registry MODULES test_registry)
py_test_modules(test_fetch_var MODULES test_fetch_var)
py_test_modules(test_dynrnn_static_input MODULES test_dynrnn_static_input)
py_test_modules(test_parallel_op MODULES test_parallel_op)
py_test_modules(test_dist_train MODULES test_dist_train)
