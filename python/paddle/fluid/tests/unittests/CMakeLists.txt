file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
set(GC_ENVS FLAGS_eager_delete_tensor_gb=0.0 FLAGS_fast_eager_deletion_mode=1 FLAGS_memory_fraction_of_eager_deletion=1.0) 

if(NOT WITH_DISTRIBUTE)
    list(REMOVE_ITEM TEST_OPS test_recv_op)
    list(REMOVE_ITEM TEST_OPS test_dist_transpiler)
    list(REMOVE_ITEM TEST_OPS test_simple_dist_transpiler)
    list(REMOVE_ITEM TEST_OPS test_listen_and_serv_op)
    LIST(REMOVE_ITEM TEST_OPS test_dist_mnist)
    LIST(REMOVE_ITEM TEST_OPS test_dist_word2vec)
    LIST(REMOVE_ITEM TEST_OPS test_dist_ctr)
    LIST(REMOVE_ITEM TEST_OPS test_dist_simnet_bow)
    LIST(REMOVE_ITEM TEST_OPS test_dist_mnist_batch_merge)
    LIST(REMOVE_ITEM TEST_OPS test_dist_text_classification)
    LIST(REMOVE_ITEM TEST_OPS test_nce_remote_table_op)
    LIST(REMOVE_ITEM TEST_OPS test_hsigmoid_remote_table_op)
endif(NOT WITH_DISTRIBUTE)

if (NOT ${WITH_GPU})
    LIST(REMOVE_ITEM TEST_OPS test_conv2d_fusion_op)
    LIST(REMOVE_ITEM TEST_OPS test_parallel_dygraph_mnist) # TODO(Yancey1989): parallel dygraph support CPU device in future
elseif(${CUDNN_VERSION} VERSION_LESS 7100)
    LIST(REMOVE_ITEM TEST_OPS test_conv2d_fusion_op)
endif()

list(REMOVE_ITEM TEST_OPS test_seq_concat_op) # FIXME(helin): https://github.com/PaddlePaddle/Paddle/issues/8290
list(REMOVE_ITEM TEST_OPS test_modified_huber_loss_op) # FIXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/5184
list(REMOVE_ITEM TEST_OPS test_lstm_unit_op) # # FIXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/5185
list(REMOVE_ITEM TEST_OPS test_cond_op) # FIXME(qijun): https://github.com/PaddlePaddle/Paddle/issues/5101#issuecomment-339814957

list(REMOVE_ITEM TEST_OPS op_test) # op_test is a helper python file, not a test
list(REMOVE_ITEM TEST_OPS decorator_helper) # decorator_helper is a helper python file, not a test
if(APPLE)
    if(NOT WITH_DISTRIBUTE)
        list(REMOVE_ITEM TEST_OPS test_desc_clone)
        list(REMOVE_ITEM TEST_OPS test_program_code)
    endif(NOT WITH_DISTRIBUTE)
    message(WARNING "These tests has been disabled in OSX before being fixed:\n test_fuse_elewise_add_act_pass \n test_detection_map_op \n test_dist_se_resnext")
    # this op is not support on mac
    list(REMOVE_ITEM TEST_OPS test_fusion_seqexpand_concat_fc_op)
    # TODO: add the unitest back when it fixed
    list(REMOVE_ITEM TEST_OPS test_detection_map_op)
    list(REMOVE_ITEM TEST_OPS test_dist_se_resnext)
    # TODO(tangwei12): add the unitest back when it fixed
    list(REMOVE_ITEM TEST_OPS test_dist_word2vec)
    list(REMOVE_ITEM TEST_OPS test_fuse_elewise_add_act_pass)
endif()
if(NOT WITH_MKLML)
    # this op is not support on openblas
    list(REMOVE_ITEM TEST_OPS test_fusion_seqexpand_concat_fc_op)
endif()

function(py_test_modules TARGET_NAME)
  if(WITH_TESTING)
    set(options SERIAL)
    set(oneValueArgs "")
    set(multiValueArgs MODULES DEPS ENVS)
    cmake_parse_arguments(py_test_modules "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_test(NAME ${TARGET_NAME}
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python ${py_test_modules_ENVS}
        ${PYTHON_EXECUTABLE} ${PADDLE_SOURCE_DIR}/tools/test_runner.py ${py_test_modules_MODULES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    if (py_test_modules_SERIAL)
        set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
    endif()
    set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT 600)
  endif()
endfunction()
list(REMOVE_ITEM TEST_OPS test_warpctc_op)
list(REMOVE_ITEM TEST_OPS test_dist_train)
list(REMOVE_ITEM TEST_OPS test_dist_transpiler)
list(REMOVE_ITEM TEST_OPS test_parallel_executor_crf)
list(REMOVE_ITEM TEST_OPS test_parallel_executor_fetch_feed)
list(REMOVE_ITEM TEST_OPS test_dist_se_resnext)
list(REMOVE_ITEM TEST_OPS test_dgc_op)
list(REMOVE_ITEM TEST_OPS test_dist_se_resnext_nccl)
list(REMOVE_ITEM TEST_OPS test_dist_transformer)
list(REMOVE_ITEM TEST_OPS test_parallel_executor_transformer)
list(REMOVE_ITEM TEST_OPS test_bilinear_interp_op)
list(REMOVE_ITEM TEST_OPS test_nearest_interp_op)
list(REMOVE_ITEM TEST_OPS test_imperative_resnet)
list(REMOVE_ITEM TEST_OPS test_imperative_se_resnext)
list(REMOVE_ITEM TEST_OPS test_imperative_mnist)
list(REMOVE_ITEM TEST_OPS test_ir_memory_optimize_transformer)
list(REMOVE_ITEM TEST_OPS test_layers)

# Some ops need to check results when gc is enabled
# Currently, only ops that register NoNeedBufferVarsInference need to do this test   
set(TEST_OPS_WITH_GC 
  test_concat_op
  test_elementwise_add_op
  test_elementwise_sub_op
  test_fill_constant_batch_size_like_op
  test_fill_zeros_like2_op
  test_gather_op
  test_gaussian_random_batch_size_like_op
  test_linear_chain_crf_op
  test_lod_reset_op
  test_lookup_table_op
  test_mean_op
  test_pad2d_op
  test_scatter_op
  test_sequence_concat
  test_seq_conv
  test_seq_pool
  test_sequence_expand_as
  test_sequence_expand
  test_sequence_pad_op
  test_sequence_unpad_op
  test_sequence_scatter_op
  test_sequence_slice_op
  test_slice_op
  test_space_to_depth_op
  test_squared_l2_distance_op
  test_uniform_random_batch_size_like_op)

foreach(TEST_OP ${TEST_OPS_WITH_GC})
  list(REMOVE_ITEM TEST_OPS ${TEST_OP})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP} ENVS ${GC_ENVS})
endforeach()

foreach(TEST_OP ${TEST_OPS})
    py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach(TEST_OP)
py_test_modules(test_adam_op_multi_thread MODULES test_adam_op ENVS FLAGS_inner_op_parallelism=4)
py_test_modules(test_warpctc_op MODULES test_warpctc_op ENVS FLAGS_warpctc_dir=${WARPCTC_LIB_DIR} SERIAL)
py_test_modules(test_bilinear_interp_op MODULES test_bilinear_interp_op ENVS ${GC_ENVS} SERIAL)
py_test_modules(test_nearest_interp_op MODULES test_nearest_interp_op ENVS ${GC_ENVS} SERIAL)
py_test_modules(test_imperative_resnet MODULES test_imperative_resnet ENVS
    FLAGS_cudnn_deterministic=1 SERIAL)
py_test_modules(test_imperative_mnist MODULES test_imperative_mnist ENVS
    FLAGS_cudnn_deterministic=1 SERIAL)
py_test_modules(test_imperative_se_resnext MODULES test_imperative_se_resnext ENVS
    FLAGS_cudnn_deterministic=1 SERIAL)

if(WITH_DISTRIBUTE)
    py_test_modules(test_dist_train MODULES test_dist_train SERIAL)
    set_tests_properties(test_listen_and_serv_op PROPERTIES TIMEOUT 20)
    if(WITH_DGC)
        py_test_modules(test_dgc_op MODULES test_dgc_op)
    endif()
    if(NOT APPLE)
        set_tests_properties(test_dist_mnist PROPERTIES TIMEOUT 200)
        set_tests_properties(test_dist_word2vec PROPERTIES TIMEOUT 200)
        py_test_modules(test_dist_se_resnext MODULES test_dist_se_resnext)
        set_tests_properties(test_dist_se_resnext PROPERTIES TIMEOUT 1000)
        py_test_modules(test_dist_se_resnext_nccl MODULES test_dist_se_resnext_nccl SERIAL)
        set_tests_properties(test_dist_se_resnext_nccl PROPERTIES TIMEOUT 1000)
        # FIXME(typhoonzero): add these tests back
        # py_test_modules(test_dist_transformer MODULES test_dist_transformer)
        # set_tests_properties(test_dist_transformer PROPERTIES TIMEOUT 1000)
        set_tests_properties(test_dist_ctr test_dist_mnist test_dist_mnist_batch_merge test_dist_save_load test_dist_se_resnext test_dist_simnet_bow test_dist_text_classification test_dist_train test_dist_word2vec PROPERTIES RUN_SERIAL TRUE)
    endif(NOT APPLE)
    # py_test_modules(test_dist_transpiler MODULES test_dist_transpiler)
endif()

py_test_modules(test_parallel_executor_crf MODULES test_parallel_executor_crf SERIAL)
py_test_modules(test_parallel_executor_fetch_feed MODULES test_parallel_executor_fetch_feed SERIAL)
set_tests_properties(test_parallel_executor_fetch_feed PROPERTIES TIMEOUT 450)
set_tests_properties(test_parallel_executor_seresnext PROPERTIES TIMEOUT 740)
py_test_modules(test_parallel_executor_transformer MODULES test_parallel_executor_transformer SERIAL)
py_test_modules(test_layers MODULES test_layers ENVS FLAGS_cudnn_deterministic=1)
if(NOT WIN32)
    py_test_modules(test_ir_memory_optimize_transformer MODULES test_ir_memory_optimize_transformer SERIAL)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # change the timeout from 600 to 2200, because in debug mode, this test need more time.
    set_tests_properties(test_parallel_executor_seresnext PROPERTIES TIMEOUT 2200)
endif()

if (WITH_NGRAPH)
    add_subdirectory(ngraph)
endif()

if (WITH_MKLDNN)
    add_subdirectory(mkldnn)
endif()

if(WITH_DISTRIBUTE)
    set_tests_properties(test_listen_and_serv_op test_nce_remote_table_op test_hsigmoid_remote_table_op
            PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
endif()

set_tests_properties(test_weight_decay test_conv_shift_op test_alloc_continuous_space_op test_recordio_reader
        test_parallel_executor_test_while_train test_adam_op_multi_thread test_parallel_executor_mnist
        test_parallel_executor_seresnext test_parallel_executor_crf test_nearest_interp_op
        PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
