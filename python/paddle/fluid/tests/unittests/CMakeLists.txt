file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

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
elseif(${CUDNN_VERSION} VERSION_LESS 7100)
    LIST(REMOVE_ITEM TEST_OPS test_conv2d_fusion_op)
endif()

list(REMOVE_ITEM TEST_OPS test_seq_concat_op) # FIXME(helin): https://github.com/PaddlePaddle/Paddle/issues/8290
list(REMOVE_ITEM TEST_OPS test_modified_huber_loss_op) # FIXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/5184
list(REMOVE_ITEM TEST_OPS test_lstm_unit_op) # # FIXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/5185
list(REMOVE_ITEM TEST_OPS test_recurrent_op) # FIXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/6152
list(REMOVE_ITEM TEST_OPS test_cond_op) # FIXME(qijun): https://github.com/PaddlePaddle/Paddle/issues/5101#issuecomment-339814957

list(REMOVE_ITEM TEST_OPS op_test) # op_test is a helper python file, not a test
list(REMOVE_ITEM TEST_OPS decorators) # decorators is a helper python file, not a test
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
list(REMOVE_ITEM TEST_OPS test_image_classification_resnet)
list(REMOVE_ITEM TEST_OPS test_bilinear_interp_op)
list(REMOVE_ITEM TEST_OPS test_nearest_interp_op)
list(REMOVE_ITEM TEST_OPS test_imperative_resnet)
list(REMOVE_ITEM TEST_OPS test_imperative_mnist)
list(REMOVE_ITEM TEST_OPS test_ir_memory_optimize_transformer)
list(REMOVE_ITEM TEST_OPS test_layers)
foreach(TEST_OP ${TEST_OPS})
    py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach(TEST_OP)
py_test_modules(test_adam_op_multi_thread MODULES test_adam_op ENVS FLAGS_inner_op_parallelism=4)
py_test_modules(test_warpctc_op MODULES test_warpctc_op ENVS FLAGS_warpctc_dir=${WARPCTC_LIB_DIR} SERIAL)
py_test_modules(test_bilinear_interp_op MODULES test_bilinear_interp_op SERIAL)
py_test_modules(test_nearest_interp_op MODULES test_nearest_interp_op SERIAL)
py_test_modules(test_imperative_resnet MODULES test_imperative_resnet ENVS
  FLAGS_cudnn_deterministic=1)
py_test_modules(test_imperative_mnist MODULES test_imperative_mnist ENVS
  FLAGS_cudnn_deterministic=1)
if(WITH_DISTRIBUTE)
    py_test_modules(test_dist_train MODULES test_dist_train SERIAL)
    set_tests_properties(test_listen_and_serv_op PROPERTIES TIMEOUT 20)
    if(NOT APPLE)
        set_tests_properties(test_dist_mnist PROPERTIES TIMEOUT 200)
        set_tests_properties(test_dist_word2vec PROPERTIES TIMEOUT 200)
        py_test_modules(test_dist_se_resnext MODULES test_dist_se_resnext)
        py_test_modules(test_dgc_op MODULES test_dgc_op)
        set_tests_properties(test_dist_se_resnext PROPERTIES TIMEOUT 1000)
        py_test_modules(test_dist_se_resnext_nccl MODULES test_dist_se_resnext_nccl)
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
py_test_modules(test_parallel_executor_transformer MODULES test_parallel_executor_transformer SERIAL)
py_test_modules(test_layers MODULES test_layers ENVS FLAGS_cudnn_deterministic=1)
if(NOT WIN32)
    py_test_modules(test_ir_memory_optimize_transformer MODULES test_ir_memory_optimize_transformer SERIAL)
endif()

if(NOT APPLE)
    py_test_modules(test_image_classification_resnet MODULES test_image_classification_resnet SERIAL)
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
