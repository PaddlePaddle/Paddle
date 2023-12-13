file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
set(GC_ENVS FLAGS_eager_delete_tensor_gb=0.0 FLAGS_fast_eager_deletion_mode=1
            FLAGS_memory_fraction_of_eager_deletion=1.0)
set(dist_ENVS http_proxy="" https_proxy="")

file(
  GLOB DIST_TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_dist_*.py")
list(REMOVE_ITEM DIST_TEST_OPS "test_dist_op")

string(REPLACE ".py" "" DIST_TEST_OPS "${DIST_TEST_OPS}")

if((NOT WITH_GPU) AND (NOT WITH_XPU))
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_mnist_batch_merge")
endif()
list(APPEND DIST_TEST_OPS test_parallel_dygraph_dataparallel)
list(APPEND DIST_TEST_OPS test_static_model_parallel_fused_feedforward)
list(APPEND DIST_TEST_OPS test_static_model_parallel_fused_attention)
list(APPEND DIST_TEST_OPS test_static_model_parallel_fused_multi_transformer)
list(APPEND DIST_TEST_OPS test_auto_parallel_data_unshard)
list(APPEND DIST_TEST_OPS test_auto_parallel_save_load)
list(APPEND DIST_TEST_OPS test_auto_parallel_autoconvert)
set(MIXED_DIST_TEST_OPS ${DIST_TEST_OPS})
#remove distribute unittests.

list(APPEND MIXED_DIST_TEST_OPS test_simple_dist_transpiler)
list(APPEND MIXED_DIST_TEST_OPS test_communicator_async)
list(APPEND MIXED_DIST_TEST_OPS test_communicator_ps_gpu)
list(APPEND MIXED_DIST_TEST_OPS test_communicator_geo)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_launch_ps)
list(APPEND MIXED_DIST_TEST_OPS test_launch_coverage)
list(APPEND MIXED_DIST_TEST_OPS test_fleetrun)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_run_random_port)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_launch_async)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_launch_cloud)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_launch_ascend)
list(APPEND MIXED_DIST_TEST_OPS test_ascend_group)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_launch_nproc)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_api_input)

list(APPEND MIXED_DIST_TEST_OPS test_fleet_base)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_base_2)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_base_3)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_auto)
list(APPEND MIXED_DIST_TEST_OPS test_auto_parallel_partitioner)
list(APPEND MIXED_DIST_TEST_OPS test_auto_parallel_partitioner_gpt)
list(APPEND MIXED_DIST_TEST_OPS test_auto_parallel_searcher)
list(APPEND MIXED_DIST_TEST_OPS test_auto_parallel_reshard)
list(APPEND MIXED_DIST_TEST_OPS test_auto_parallel_dist_tensor)
list(APPEND MIXED_DIST_TEST_OPS test_auto_parallel_reshard_serial)
list(APPEND MIXED_DIST_TEST_OPS test_auto_parallel_reshard_mppp)
list(APPEND MIXED_DIST_TEST_OPS test_auto_parallel_reshard_dpmppp)
list(APPEND MIXED_DIST_TEST_OPS test_auto_parallel_cost_model)
list(APPEND MIXED_DIST_TEST_OPS test_dygraph_hybrid_dp)
foreach(TEST_OP ${MIXED_DIST_TEST_OPS})
  list(REMOVE_ITEM TEST_OPS ${TEST_OP})
endforeach()

if(NOT WITH_PYTHON AND ON_INFER)
  list(REMOVE_ITEM TEST_OPS test_eager_trace_op)
endif()

if(NOT WITH_GPU)
  list(REMOVE_ITEM TEST_OPS test_async_read_write)
  list(REMOVE_ITEM TEST_OPS test_fused_feedforward_op)
  list(REMOVE_ITEM TEST_OPS test_fused_attention_op)
  list(REMOVE_ITEM TEST_OPS test_fused_attention_op_api)
  list(REMOVE_ITEM TEST_OPS test_fused_multi_transformer_op)
  list(REMOVE_ITEM TEST_OPS test_fused_multi_transformer_int8_op)
  list(REMOVE_ITEM TEST_OPS test_fused_transformer_encoder_layer)
  list(REMOVE_ITEM TEST_OPS test_fused_bias_dropout_residual_layer_norm_op)
  list(REMOVE_ITEM TEST_OPS test_fused_bias_dropout_residual_layer_norm_op_api)
  list(REMOVE_ITEM TEST_OPS test_rms_norm_op)
  list(REMOVE_ITEM TEST_OPS test_fused_layernorm_op)
  list(REMOVE_ITEM TEST_OPS test_fused_attention_pass)
  list(REMOVE_ITEM TEST_OPS test_fused_feedforward_pass)
  list(REMOVE_ITEM TEST_OPS test_fused_comm_buffer)
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_hapi_model")
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_fleet_spmt")
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_fleet_minimize")
  list(REMOVE_ITEM TEST_OPS test_async_read_write)
endif()

list(REMOVE_ITEM TEST_OPS test_audio_logmel_feature test_audio_mel_feature)
list(REMOVE_ITEM TEST_OPS test_fused_ec_moe_op)
list(REMOVE_ITEM TEST_OPS test_fused_gemm_epilogue_op)
list(REMOVE_ITEM TEST_OPS test_fused_gemm_epilogue_grad_op)
list(REMOVE_ITEM TEST_OPS test_fuse_gemm_epilogue_pass)
list(REMOVE_ITEM TEST_OPS test_fused_dot_product_attention_op)
list(REMOVE_ITEM TEST_OPS test_fuse_dot_product_attention_pass)

if(((NOT WITH_ROCM) AND (NOT WITH_GPU)) OR WIN32)
  list(REMOVE_ITEM TEST_OPS test_c_comm_init_all_op)
  list(REMOVE_ITEM TEST_OPS test_c_embedding_op)
  list(REMOVE_ITEM TEST_OPS test_pipeline_parallel)
  list(REMOVE_ITEM TEST_OPS test_fleet_perf_test)
  list(REMOVE_ITEM TEST_OPS test_memcpy_op)
  list(REMOVE_ITEM TEST_OPS test_raw_program_optimizer)
  list(REMOVE_ITEM TEST_OPS test_fleet_gradient_scale)
  list(REMOVE_ITEM TEST_OPS test_disable_signal_handler)
  list(REMOVE_ITEM TEST_OPS test_fleet_executor)
  list(REMOVE_ITEM TEST_OPS test_fleet_executor_with_task_nodes)
  list(REMOVE_ITEM TEST_OPS test_fleet_executor_multi_devices)
  list(REMOVE_ITEM TEST_OPS test_fleet_executor_origin_scheduler)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_mapper)
  list(REMOVE_ITEM TEST_OPS test_fleet_executor_task_node)
  list(REMOVE_ITEM TEST_OPS test_fleet_exe_dist_model_run)
  list(REMOVE_ITEM TEST_OPS test_fleet_exe_dist_model_tensor)
  list(REMOVE_ITEM TEST_OPS test_fleet_executor_cond_interceptor)
endif()

list(REMOVE_ITEM TEST_OPS test_deprecated_decorator)

if(WIN32)
  list(REMOVE_ITEM TEST_OPS test_multiprocess_reader_exception)
  list(REMOVE_ITEM TEST_OPS test_trainer_desc)
  list(REMOVE_ITEM TEST_OPS test_checkpoint_notify_op)
  list(REMOVE_ITEM TEST_OPS test_downpoursgd)
  list(REMOVE_ITEM TEST_OPS test_fleet)
  list(REMOVE_ITEM TEST_OPS test_fleet_nocvm_1)
  list(REMOVE_ITEM TEST_OPS test_fleet_rolemaker)
  list(REMOVE_ITEM TEST_OPS test_fleet_rolemaker_3)
  list(REMOVE_ITEM TEST_OPS test_fleet_unitaccessor)
  list(REMOVE_ITEM TEST_OPS test_ps_dispatcher)
  list(REMOVE_ITEM TEST_OPS test_ir_memory_optimize_nlp)
  list(REMOVE_ITEM TEST_OPS test_nvprof)

  # TODO: Fix these unittests failed on Windows
  list(REMOVE_ITEM TEST_OPS test_debugger)
  if(WITH_GPU)
    list(REMOVE_ITEM TEST_OPS test_update_loss_scaling_op)
  endif()
endif()

if(NOT WITH_DISTRIBUTE OR WIN32)
  # DISTRIBUTE related
  list(REMOVE_ITEM TEST_OPS test_avoid_twice_initialization)
  list(REMOVE_ITEM TEST_OPS test_fleet_metric)
  list(REMOVE_ITEM TEST_OPS test_fleet_ps)
  list(REMOVE_ITEM TEST_OPS test_fleet_rolemaker_2)
  list(REMOVE_ITEM TEST_OPS test_delete_c_identity_op_pass)
  # TODO: Fix these unittests failed on Windows
  list(REMOVE_ITEM TEST_OPS test_fake_init_op)
endif()

if(NOT WITH_DISTRIBUTE)
  list(REMOVE_ITEM TEST_OPS test_desc_clone_dist)
endif()

if(WIN32)
  list(REMOVE_ITEM TEST_OPS test_complex_matmul)
  list(REMOVE_ITEM TEST_OPS test_ops_nms)
  list(REMOVE_ITEM TEST_OPS test_trt_convert_preln_residual_bias)
  list(REMOVE_ITEM TEST_OPS test_fused_multi_transformer_int8_op)
  list(REMOVE_ITEM TEST_OPS test_masked_multihead_attention_op)
  list(REMOVE_ITEM TEST_OPS test_fused_ec_moe_op)
  list(REMOVE_ITEM TEST_OPS test_rms_norm_op)
  list(REMOVE_ITEM TEST_OPS test_fused_layernorm_op)
  list(REMOVE_ITEM TEST_OPS test_matmul_int8_op)
  list(REMOVE_ITEM TEST_OPS test_variable_length_memory_efficient_attention)
endif()
list(REMOVE_ITEM TEST_OPS test_checkpoint_saver)

if(APPLE OR WIN32)
  list(REMOVE_ITEM TEST_OPS test_fs_interface)
  list(REMOVE_ITEM TEST_OPS test_fleet_metric)
endif()

list(REMOVE_ITEM TEST_OPS test_parallel_dygraph_hybrid_parallel)

list(REMOVE_ITEM TEST_OPS test_parallel_dygraph_transformer_gloo)
# NOTE: @xiongkun03, cpu is too slow, fix it in next PR

if(NOT WITH_GLOO)
  list(REMOVE_ITEM TEST_OPS test_parallel_dygraph_dataparallel_cpuonly)

  list(REMOVE_ITEM TEST_OPS test_parallel_dygraph_unused_variables_gloo)
  list(REMOVE_ITEM TEST_OPS
       test_parallel_dygraph_sparse_embedding_over_height_gloo)
  list(REMOVE_ITEM TEST_OPS test_parallel_dygraph_sparse_embedding_gloo)
  list(REMOVE_ITEM TEST_OPS
       test_parallel_dygraph_sparse_embedding_diff_length_gloo)
endif()

if((NOT WITH_GPU) AND (NOT WITH_ROCM))
  list(REMOVE_ITEM TEST_OPS test_fused_conv2d_add_act_op)
  list(REMOVE_ITEM TEST_OPS test_rank_attention_op)
  # TODO(shenliang03): rank_attention_op support CPU device in future
  list(REMOVE_ITEM TEST_OPS test_batch_fc_op)
  # TODO(shenliang03): batch_fc_op support CPU device in future
  # TODO(Yancey1989): parallel dygraph support CPU device in future
  list(REMOVE_ITEM TEST_OPS test_parallel_dygraph_dataparallel)
  list(REMOVE_ITEM TEST_OPS test_fleet_base_single)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_partitioner)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_partitioner_gpt)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_searcher)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_reshard)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_dist_tensor)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_reshard_serial)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_reshard_mppp)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_reshard_dpmppp)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_cost_model)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_data_unshard)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_save_load)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_autoconvert)

elseif(WITH_GPU)
  if(${CUDNN_VERSION} VERSION_LESS 7100)
    list(REMOVE_ITEM TEST_OPS test_fused_conv2d_add_act_op)
  endif()
endif()

if((NOT WITH_NCCL) AND (NOT WITH_RCCL))
  list(REMOVE_ITEM TEST_OPS test_imperative_group)
endif()

if(((NOT WITH_ROCM) AND (NOT WITH_GPU)) OR WIN32)
  list(REMOVE_ITEM TEST_OPS test_fused_gate_attention_op)
  list(REMOVE_ITEM TEST_OPS test_boxps)
  list(REMOVE_ITEM TEST_OPS test_allgather)
  list(REMOVE_ITEM TEST_OPS test_reducescatter)
  list(REMOVE_ITEM TEST_OPS test_reducescatter_api)
endif()
list(REMOVE_ITEM TEST_OPS test_seq_concat_op)
# FIXME(helin): https://github.com/PaddlePaddle/Paddle/issues/8290
list(REMOVE_ITEM TEST_OPS test_lstm_unit_op)
# # FIXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/5185
list(REMOVE_ITEM TEST_OPS test_cond_op)

# FIXME(qijun): https://github.com/PaddlePaddle/Paddle/issues/5101#issuecomment-339814957

list(REMOVE_ITEM TEST_OPS op_test) # op_test is a helper python file, not a test
list(REMOVE_ITEM TEST_OPS decorator_helper)
# decorator_helper is a helper python file, not a test

if(APPLE)
  if(NOT WITH_DISTRIBUTE)
    list(REMOVE_ITEM TEST_OPS test_desc_clone)
    list(REMOVE_ITEM TEST_OPS test_program_code)
  endif()
  message(
    WARNING
      "These tests has been disabled in OSX before being fixed:\n test_fuse_elewise_add_act_pass \n test_detection_map_op \n test_dist_se_resnext_*"
  )
  # this op is not support on mac
  list(REMOVE_ITEM TEST_OPS test_fusion_seqexpand_concat_fc_op)
  list(REMOVE_ITEM TEST_OPS test_detection_map_op)
  list(REMOVE_ITEM TEST_OPS test_fuse_elewise_add_act_pass)
endif()
if(NOT WITH_MKLML)
  # this op is not support on openblas
  list(REMOVE_ITEM TEST_OPS test_fusion_seqexpand_concat_fc_op)
endif()

if(NOT WITH_MKL OR NOT WITH_AVX)
  list(REMOVE_ITEM TEST_OPS test_match_matrix_tensor_op)
  list(REMOVE_ITEM TEST_OPS test_var_conv_2d)
endif()

if(WITH_COVERAGE
   OR WIN32
   OR WITH_NV_JETSON)
  list(REMOVE_ITEM TEST_OPS test_pyramid_hash_op)
endif()

list(REMOVE_ITEM TEST_OPS test_fleet_pyramid_hash)

if((WITH_ROCM OR WITH_GPU) OR NOT WITH_MKLML)
  # matmul with multiple heads need MKL support
  list(REMOVE_ITEM TEST_OPS test_matmul_op_with_head)
endif()

if(NOT WITH_CRYPTO)
  list(REMOVE_ITEM TEST_OPS test_crypto)
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

function(parallel_bash_test_modules TARGET_NAME)
  if(NOT WITH_TESTING)
    return()
  endif()

  set(options SERIAL)
  set(oneValueArgs TIMEOUT START_BASH)
  set(multiValueArgs DEPS ENVS LABELS UnitTests)
  cmake_parse_arguments(parallel_bash_test_modules "${options}"
                        "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(timeout 120)
  if(${parallel_bash_test_modules_TIMEOUT})
    set(timeout ${parallel_bash_test_modules_TIMEOUT})
  endif()

  list(JOIN parallel_bash_test_modules_UnitTests " " uts_string)

  if(WITH_COVERAGE)
    add_test(
      NAME ${TARGET_NAME}
      COMMAND
        ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python
        TEST_TARGET_NAME=${TARGET_NAME} TEST_TIMEOUT=${timeout}
        ${parallel_bash_test_modules_ENVS} UnitTests=${uts_string}
        WITH_COVERAGE=ON COVERAGE_FILE=${PADDLE_BINARY_DIR}/python-coverage.data
        bash
        ${CMAKE_CURRENT_BINARY_DIR}/${parallel_bash_test_modules_START_BASH}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  else()
    add_test(
      NAME ${TARGET_NAME}
      COMMAND
        ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python
        TEST_TARGET_NAME=${TARGET_NAME} TEST_TIMEOUT=${timeout}
        ${parallel_bash_test_modules_ENVS} UnitTests=${uts_string} bash
        ${CMAKE_CURRENT_BINARY_DIR}/${parallel_bash_test_modules_START_BASH}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()

  if(parallel_bash_test_modules_SERIAL)
    set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
  endif()

  if(parallel_bash_test_modules_LABELS)
    set_tests_properties(${TARGET_NAME}
                         PROPERTIES LABELS ${parallel_bash_test_modules_LABELS})
  endif()
endfunction()

list(REMOVE_ITEM TEST_OPS test_feed_data_check_shape_type)
list(REMOVE_ITEM TEST_OPS test_fetch_lod_tensor_array)
list(REMOVE_ITEM TEST_OPS test_warpctc_op)
list(REMOVE_ITEM TEST_OPS test_data_norm_op)
list(REMOVE_ITEM TEST_OPS test_parallel_executor_transformer)
list(REMOVE_ITEM TEST_OPS test_parallel_executor_transformer_auto_growth)
list(REMOVE_ITEM TEST_OPS test_bilinear_interp_op)
list(REMOVE_ITEM TEST_OPS test_nearest_interp_op)
list(REMOVE_ITEM TEST_OPS test_imperative_resnet)
list(REMOVE_ITEM TEST_OPS test_imperative_resnet_sorted_gradient)
list(REMOVE_ITEM TEST_OPS test_imperative_mnist_sorted_gradient)
list(REMOVE_ITEM TEST_OPS test_imperative_se_resnext)
list(REMOVE_ITEM TEST_OPS test_imperative_mnist)
list(REMOVE_ITEM TEST_OPS test_ir_memory_optimize_transformer)
list(REMOVE_ITEM TEST_OPS test_layers)
list(REMOVE_ITEM TEST_OPS test_parallel_executor_seresnext_base_cpu)
list(REMOVE_ITEM TEST_OPS test_parallel_executor_seresnext_with_reduce_cpu)
list(REMOVE_ITEM TEST_OPS
     test_parallel_executor_seresnext_with_fuse_all_reduce_cpu)
list(REMOVE_ITEM TEST_OPS test_imperative_ocr_attention_model)
list(REMOVE_ITEM TEST_OPS test_async_ssa_graph_executor_mnist)
list(REMOVE_ITEM TEST_OPS test_install_check)
list(REMOVE_ITEM TEST_OPS test_basic_gru_api)
list(REMOVE_ITEM TEST_OPS test_basic_gru_unit_op)
list(REMOVE_ITEM TEST_OPS test_basic_lstm_api)
list(REMOVE_ITEM TEST_OPS test_basic_lstm_unit_op)
list(REMOVE_ITEM TEST_OPS test_fuse_all_reduce_pass)
list(REMOVE_ITEM TEST_OPS test_fuse_bn_act_pass)
list(REMOVE_ITEM TEST_OPS test_fuse_bn_add_act_pass)
list(REMOVE_ITEM TEST_OPS test_conv3d_transpose_op)

# disable this unittest temporarily
list(REMOVE_ITEM TEST_OPS test_imperative_data_loader_exception)

# disable sparse_attention which not in suitable env
if((NOT WITH_GPU)
   OR (WIN32)
   OR (PADDLE_WITH_ARM)
   OR (WITH_ROCM))
  list(REMOVE_ITEM TEST_OPS test_sparse_attention_op)
endif()

if(APPLE OR WIN32)
  list(REMOVE_ITEM TEST_OPS test_dataset)
  list(REMOVE_ITEM TEST_OPS test_dataset_dataloader)
  list(REMOVE_ITEM TEST_OPS test_imperative_data_loader_base)
  list(REMOVE_ITEM TEST_OPS test_imperative_data_loader_process)
  list(REMOVE_ITEM TEST_OPS test_imperative_data_loader_fds_clear)
  list(REMOVE_ITEM TEST_OPS test_imperative_data_loader_exit_func)
  list(REMOVE_ITEM TEST_OPS test_imperative_signal_handler)
  list(REMOVE_ITEM TEST_OPS test_multiprocess_dataloader_static)
  list(REMOVE_ITEM TEST_OPS test_multiprocess_dataloader_dynamic)
  list(REMOVE_ITEM TEST_OPS test_multiprocess_dataloader_exception)
  list(REMOVE_ITEM TEST_OPS test_multiprocess_dataloader_iterable_dataset)
  list(REMOVE_ITEM TEST_OPS test_multiprocess_dataloader_dataset)
  list(REMOVE_ITEM TEST_OPS test_paddle_multiprocessing)
endif()

if(NOT WITH_GLOO)
  list(REMOVE_ITEM TEST_OPS test_cpuonly_spawn)
endif()

if(NOT WITH_GPU
   OR WIN32
   OR APPLE)
  list(REMOVE_ITEM TEST_OPS test_build_strategy_fusion_group_pass)
endif()

if(NOT WITH_CUDNN_FRONTEND)
  list(REMOVE_ITEM TEST_OPS test_fused_scale_bias_relu_conv_bn_op)
  list(REMOVE_ITEM TEST_OPS test_fused_scale_bias_add_relu_op)
  list(REMOVE_ITEM TEST_OPS test_fused_dconv_drelu_dbn_op)
endif()

# Some ops need to check results when gc is enabled
# Currently, only ops that register NoNeedBufferVarsInference need to do this test
set(TEST_OPS_WITH_GC
    test_affine_channel_op
    test_concat_op
    test_elementwise_add_op
    test_elementwise_sub_op
    test_fill_zeros_like2_op
    test_gather_op
    test_gather_nd_op
    test_linear_chain_crf_op
    test_lod_reset_op
    test_lookup_table_op
    test_mean_op
    test_scatter_op
    test_slice_op)

foreach(TEST_OP ${TEST_OPS_WITH_GC})
  list(REMOVE_ITEM TEST_OPS ${TEST_OP})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP} ENVS ${GC_ENVS})
endforeach()

# Switch some dy2st UT to eager mode
set(TEST_EAGER_OPS test_jit_save_load test_translated_layer)
foreach(TEST_OP ${TEST_EAGER_OPS})
  list(REMOVE_ITEM TEST_OPS ${TEST_OP})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP} ENVS FLAGS_enable_eager_mode=1)
endforeach()

if((NOT WITH_GPU)
   AND (NOT WITH_XPU)
   AND NOT (WITH_ASCEND OR WITH_ASCEND_CL))
  list(REMOVE_ITEM TEST_OPS "test_fleet_graph_execution_meta_optimizer")
  list(REMOVE_ITEM TEST_OPS "test_dist_fleet_grad_clip")
  list(REMOVE_ITEM TEST_OPS "test_dist_fleet_heter_ctr")
  list(REMOVE_ITEM TEST_OPS "test_dist_fleet_ps_gpu_ctr")
  list(REMOVE_ITEM TEST_OPS "test_dist_mnist_batch_merge")
endif()

list(REMOVE_ITEM TEST_OPS "test_stride")
if(WITH_COVERAGE)
  list(REMOVE_ITEM TEST_OPS test_weight_decay)
  list(REMOVE_ITEM TEST_OPS test_graph_reindex)
  list(REMOVE_ITEM TEST_OPS test_cuda_graphed_layer)
  list(REMOVE_ITEM TEST_OPS test_cuda_graph_partial_graph_static_run)
  list(REMOVE_ITEM DIST_TEST_OPS test_dist_fleet_geo)
endif()
foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach()
set_tests_properties(test_logcumsumexp_op PROPERTIES TIMEOUT 30)
py_test_modules(test_adam_op_multi_thread MODULES test_adam_op ENVS
                FLAGS_inner_op_parallelism=4)
if(WITH_GPU
   OR WITH_XPU
   OR WITH_ASCEND
   OR WITH_ASCEND_CL
   OR APPLE)
  py_test_modules(test_warpctc_op MODULES test_warpctc_op)
  set_tests_properties(test_warpctc_op PROPERTIES TIMEOUT 120)
endif()
py_test_modules(test_bilinear_interp_op MODULES test_bilinear_interp_op ENVS
                ${GC_ENVS})
py_test_modules(test_nearest_interp_op MODULES test_nearest_interp_op ENVS
                ${GC_ENVS})
py_test_modules(test_imperative_resnet MODULES test_imperative_resnet ENVS
                FLAGS_cudnn_deterministic=1)
set_tests_properties(test_imperative_resnet
                     PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE:NIGHTLY")
py_test_modules(
  test_imperative_resnet_sorted_gradient MODULES
  test_imperative_resnet_sorted_gradient ENVS FLAGS_cudnn_deterministic=1)
set_tests_properties(test_imperative_resnet_sorted_gradient
                     PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE:NIGHTLY")
py_test_modules(test_imperative_mnist MODULES test_imperative_mnist ENVS
                FLAGS_cudnn_deterministic=1)
py_test_modules(
  test_imperative_mnist_sorted_gradient MODULES
  test_imperative_mnist_sorted_gradient ENVS FLAGS_cudnn_deterministic=1)
py_test_modules(test_imperative_se_resnext MODULES test_imperative_se_resnext
                ENVS FLAGS_cudnn_deterministic=1)
set_tests_properties(test_imperative_se_resnext
                     PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE:NIGHTLY")
py_test_modules(
  test_imperative_ocr_attention_model MODULES
  test_imperative_ocr_attention_model ENVS FLAGS_cudnn_deterministic=1)
py_test_modules(test_install_check MODULES test_install_check ENVS
                FLAGS_cudnn_deterministic=1)
set_tests_properties(test_install_check PROPERTIES LABELS "RUN_TYPE=DIST")
py_test_modules(test_install_check_pir MODULES test_install_check ENVS
                FLAGS_cudnn_deterministic=1 FLAGS_enable_pir_in_executor=1)
set_tests_properties(test_install_check_pir PROPERTIES LABELS "RUN_TYPE=DIST")

if((WITH_GPU) AND (CUDA_VERSION GREATER_EQUAL 11.6))
  py_test_modules(test_fused_gemm_epilogue_op MODULES
                  test_fused_gemm_epilogue_op)
  py_test_modules(test_fused_gemm_epilogue_grad_op MODULES
                  test_fused_gemm_epilogue_grad_op)
  py_test_modules(
    test_fused_gemm_epilogue_op_with_es MODULES test_fused_gemm_epilogue_op
    ENVS FLAGS_cublaslt_exhaustive_search_times=30)
  py_test_modules(
    test_fused_gemm_epilogue_grad_op_with_es MODULES
    test_fused_gemm_epilogue_grad_op ENVS
    FLAGS_cublaslt_exhaustive_search_times=30)
  py_test_modules(test_fuse_gemm_epilogue_pass MODULES
                  test_fuse_gemm_epilogue_pass)
endif()

if((WITH_GPU) AND (WITH_CUDNN_FRONTEND))
  py_test_modules(test_fuse_dot_product_attention_pass MODULES
                  test_fuse_dot_product_attention_pass)
  py_test_modules(test_fused_dot_product_attention_op MODULES
                  test_fused_dot_product_attention_op)
endif()

set_tests_properties(test_conv2d_op PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
set_tests_properties(test_conv2d_op_depthwise_conv
                     PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
set_tests_properties(test_conv2d_api PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
set_tests_properties(test_conv_nn_grad PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
set_tests_properties(test_norm_nn_grad PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
set_tests_properties(test_nn_grad PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
if(WITH_DISTRIBUTE)
  # FIXME(typhoonzero): add these tests back
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_transformer")
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_transpiler")

  # TODO(sandyhouse): fix and add the ut back
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_mnist_hallreduce")

  #not need
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_base")
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_fleet_base")

  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_ctr")
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_mnist_lars")
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_mnist_train")
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_save_load")
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_text_classification")
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_train")
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_word2vec")
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_fleet_gloo")

  if(NOT WITH_HETERPS)
    list(REMOVE_ITEM DIST_TEST_OPS "test_communicator_ps_gpu")
    list(REMOVE_ITEM DIST_TEST_OPS "test_dist_fleet_ps11")
    list(REMOVE_ITEM DIST_TEST_OPS "test_dist_fleet_ps12")
  endif()

  py_test_modules(test_communicator_async MODULES test_communicator_async ENVS
                  ${dist_ENVS})
  py_test_modules(test_communicator_geo MODULES test_communicator_geo ENVS
                  ${dist_ENVS})
  if(NOT APPLE)
    py_test_modules(test_fleet_base MODULES test_fleet_base ENVS ${dist_ENVS})
    py_test_modules(test_fleet_base_2 MODULES test_fleet_base_2 ENVS
                    ${dist_ENVS})
    py_test_modules(test_fleet_base_3 MODULES test_fleet_base_3 ENVS
                    ${dist_ENVS})
    if(NOT WIN32)
      py_test_modules(test_auto_parallel_partitioner MODULES
                      test_auto_parallel_partitioner ENVS ${dist_ENVS})
      py_test_modules(test_auto_parallel_partitioner_gpt MODULES
                      test_auto_parallel_partitioner_gpt ENVS ${dist_ENVS})
      py_test_modules(test_auto_parallel_searcher MODULES
                      test_auto_parallel_searcher ENVS ${dist_ENVS})
      py_test_modules(test_auto_parallel_reshard MODULES
                      test_auto_parallel_reshard ENVS ${dist_ENVS})
      py_test_modules(test_auto_parallel_dist_tensor MODULES
                      test_auto_parallel_dist_tensor ENVS ${dist_ENVS})
      py_test_modules(test_auto_parallel_reshard_serial MODULES
                      test_auto_parallel_reshard_serial ENVS ${dist_ENVS})
      py_test_modules(test_auto_parallel_reshard_mppp MODULES
                      test_auto_parallel_reshard_mppp ENVS ${dist_ENVS})
      py_test_modules(test_auto_parallel_reshard_dpmppp MODULES
                      test_auto_parallel_reshard_dpmppp ENVS ${dist_ENVS})
      py_test_modules(test_auto_parallel_cost_model MODULES
                      test_auto_parallel_cost_model ENVS ${dist_ENVS})

    endif()
  endif()

  if(NOT APPLE)
    if(WITH_GPU OR WITH_ROCM)
      py_test_modules(test_launch_coverage MODULES test_launch_coverage)
    endif()

    bash_test_modules(test_fleetrun START_BASH test_fleetrun.sh ENVS
                      PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR})
    if(WITH_GPU
       OR WITH_XPU
       OR WITH_ASCEND
       OR WITH_ASCEND_CL)
      bash_test_modules(
        test_fleet_launch_nproc START_BASH test_fleet_launch_nproc.sh ENVS
        PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR})
      bash_test_modules(
        test_fleet_run_random_port START_BASH test_fleet_run_random_port.sh
        ENVS PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR})
      bash_test_modules(
        test_fleet_launch_async START_BASH test_fleet_launch_async.sh ENVS
        PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR})
      bash_test_modules(
        test_fleet_launch_cloud START_BASH test_fleet_launch_cloud.sh ENVS
        PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR})
    endif()
    if(WITH_ASCEND OR WITH_ASCEND_CL)
      bash_test_modules(
        test_fleet_launch_ascend START_BASH test_fleet_launch_ascend.sh ENVS
        PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR})
      bash_test_modules(test_ascend_group START_BASH test_ascend_group.sh ENVS
                        PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR})
    endif()

    # port range (20000, 21200) is reserved for dist-ops
    set(dist_ut_port 20001)
    foreach(TEST_OP ${DIST_TEST_OPS})
      bash_test_modules(
        ${TEST_OP}
        START_BASH
        dist_test.sh
        LABELS
        "RUN_TYPE=EXCLUSIVE"
        ENVS
        "PADDLE_DIST_UT_PORT=${dist_ut_port}")
      math(EXPR dist_ut_port "${dist_ut_port}+10")
      if(dist_ut_port GREATER_EQUAL 21198)
        message(
          FATAL_ERROR "available ports have been exhausted:${dist_ut_port}")
      endif()
    endforeach()
    # solve it later.
    bash_test_modules(
      test_fleet_launch_ps
      START_BASH
      test_fleet_launch_ps.sh
      LABELS
      "RUN_TYPE=EXCLUSIVE"
      ENVS
      "PADDLE_DIST_UT_PORT=${dist_ut_port}"
      PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR})
    if(WITH_GLOO)
      bash_test_modules(
        test_cpuonly_launch
        START_BASH
        test_cpuonly_launch.sh
        LABELS
        "RUN_TYPE=EXCLUSIVE"
        ENVS
        "PADDLE_DIST_UT_PORT=${dist_ut_port}"
        PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR})
    endif()
  endif()
endif()

py_test_modules(test_parallel_executor_transformer MODULES
                test_parallel_executor_transformer)
if(WIN32)
  py_test_modules(
    test_parallel_executor_transformer_auto_growth MODULES
    test_parallel_executor_transformer_auto_growth ENVS
    FLAGS_allocator_strategy=auto_growth CUDA_VISIBLE_DEVICES=0)
  py_test_modules(test_fuse_all_reduce_pass MODULES test_fuse_all_reduce_pass
                  ENVS CUDA_VISIBLE_DEVICES=0)
  py_test_modules(test_feed_data_check_shape_type MODULES
                  test_feed_data_check_shape_type ENVS CUDA_VISIBLE_DEVICES=0)
  py_test_modules(test_fetch_lod_tensor_array MODULES
                  test_fetch_lod_tensor_array ENVS CUDA_VISIBLE_DEVICES=0)
else()
  py_test_modules(
    test_parallel_executor_transformer_auto_growth MODULES
    test_parallel_executor_transformer_auto_growth ENVS
    FLAGS_allocator_strategy=auto_growth)
  py_test_modules(test_fuse_all_reduce_pass MODULES test_fuse_all_reduce_pass)
  py_test_modules(test_feed_data_check_shape_type MODULES
                  test_feed_data_check_shape_type)
  py_test_modules(test_fetch_lod_tensor_array MODULES
                  test_fetch_lod_tensor_array)
endif()

py_test_modules(test_data_norm_op MODULES test_data_norm_op)
py_test_modules(
  test_fuse_bn_act_pass
  MODULES
  test_fuse_bn_act_pass
  ENVS
  FLAGS_cudnn_deterministic=1
  FLAGS_cudnn_batchnorm_spatial_persistent=1
  FLAGS_conv_workspace_size_limit=1000)
py_test_modules(
  test_fuse_bn_add_act_pass
  MODULES
  test_fuse_bn_add_act_pass
  ENVS
  FLAGS_cudnn_deterministic=1
  FLAGS_cudnn_batchnorm_spatial_persistent=1
  FLAGS_conv_workspace_size_limit=1000)

# NOTE: These unittests will appear NaN steadily in windows CI. After analysis,
# it is found that windows CI will run all the training unittests with the ON_INFER option turned on,
# which will not appear in other CIs. The calculation behavior of some ops in inference mode is
# inconsistent with that in non-inference mode.
if(WITH_PYTHON)
  py_test_modules(test_parallel_executor_seresnext_base_cpu MODULES
                  test_parallel_executor_seresnext_base_cpu)
  py_test_modules(test_parallel_executor_seresnext_with_reduce_cpu MODULES
                  test_parallel_executor_seresnext_with_reduce_cpu)
  py_test_modules(
    test_parallel_executor_seresnext_with_fuse_all_reduce_cpu MODULES
    test_parallel_executor_seresnext_with_fuse_all_reduce_cpu)
  set_tests_properties(test_parallel_executor_seresnext_base_cpu
                       PROPERTIES TIMEOUT 900)
  set_tests_properties(test_parallel_executor_seresnext_base_cpu
                       PROPERTIES LABELS "RUN_TYPE=NIGHTLY")
  set_tests_properties(test_parallel_executor_seresnext_with_reduce_cpu
                       PROPERTIES TIMEOUT 750)
  set_tests_properties(test_parallel_executor_seresnext_with_reduce_cpu
                       PROPERTIES LABELS "RUN_TYPE=NIGHTLY")
  set_tests_properties(test_parallel_executor_seresnext_with_fuse_all_reduce_cpu
                       PROPERTIES TIMEOUT 750)
  set_tests_properties(test_parallel_executor_seresnext_with_fuse_all_reduce_cpu
                       PROPERTIES LABELS "RUN_TYPE=NIGHTLY")
endif()

if(NOT WIN32)
  # TODO: fix these unittests failure on Windows
  py_test_modules(test_layers MODULES test_layers ENVS
                  FLAGS_cudnn_deterministic=1)
  py_test_modules(test_ir_memory_optimize_transformer MODULES
                  test_ir_memory_optimize_transformer)
endif()

if(WITH_HETERPS)
  set_tests_properties(test_dist_fleet_ps11 PROPERTIES LABELS "RUN_TYPE=GPUPS")
  set_tests_properties(test_dist_fleet_ps12 PROPERTIES LABELS "RUN_TYPE=GPUPS")
endif()

if(WIN32)
  cc_test(
    cc_imp_py_test
    SRCS cc_imp_py_test.cc
    DEPS python)
endif()

set_tests_properties(
  test_buffer_shared_memory_reuse_pass_and_fuse_optimization_op_pass
  test_data_norm_op test_dataloader_keep_order test_dataloader_unkeep_order
  test_buffer_shared_memory_reuse_pass PROPERTIES LABELS "RUN_TYPE=DIST")
set_tests_properties(
  test_sync_batch_norm_op
  test_parallel_executor_seresnext_base_gpu
  test_parallel_executor_seresnext_with_reduce_gpu
  test_parallel_executor_seresnext_with_fuse_all_reduce_gpu
  test_distributed_fused_lamb_op_with_clip
  test_distributed_fused_lamb_op_without_clip
  test_distributed_fused_lamb_op_with_gradient_merge
  PROPERTIES LABELS "RUN_TYPE=DIST")

if(NOT WIN32 AND NOT APPLE)
  set_tests_properties(test_imperative_signal_handler
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  set_tests_properties(test_imperative_data_loader_base
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  set_tests_properties(test_imperative_data_loader_fds_clear
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  set_tests_properties(test_multiprocess_dataloader_static
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  set_tests_properties(test_multiprocess_dataloader_dynamic
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  set_tests_properties(test_multiprocess_dataloader_exception
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  set_tests_properties(test_multiprocess_dataloader_iterable_dataset_static
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  set_tests_properties(test_multiprocess_dataloader_iterable_dataset_dynamic
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  set_tests_properties(test_multiprocess_dataloader_dataset
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  set_tests_properties(test_multiprocess_dataloader_static PROPERTIES TIMEOUT
                                                                      120)
endif()

if(NOT WIN32)
  set_tests_properties(test_multiprocess_reader_exception
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  set_tests_properties(test_layers PROPERTIES TIMEOUT 120)
  if(WITH_NV_JETSON)
    set_tests_properties(test_ir_memory_optimize_transformer PROPERTIES TIMEOUT
                                                                        1200)
  else()
    set_tests_properties(test_ir_memory_optimize_transformer PROPERTIES TIMEOUT
                                                                        120)
  endif()
endif()

if(WITH_DISTRIBUTE)
  set_tests_properties(test_dist_fleet_ctr2 PROPERTIES TIMEOUT 200)
  set_tests_properties(test_dist_fleet_sparse_embedding_ctr PROPERTIES TIMEOUT
                                                                       200)
  set_tests_properties(test_dist_fleet_infer PROPERTIES TIMEOUT 200)
  set_tests_properties(test_dist_fleet_raw_program_optimizer_fuse_allreduce
                       PROPERTIES TIMEOUT 60)
  set_tests_properties(test_dist_dygraph_apis PROPERTIES TIMEOUT 120)
endif()

# setting timeout value as 15S
set_tests_properties(test_binomial_op PROPERTIES TIMEOUT 30)
set_tests_properties(test_run PROPERTIES TIMEOUT 120)
set_tests_properties(test_sync_batch_norm_op PROPERTIES TIMEOUT 180)
set_tests_properties(test_cross_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_lod_tensor_to_selected_rows
                     PROPERTIES TIMEOUT 200)
set_tests_properties(test_lstm_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_star_gan_with_gradient_penalty
                     PROPERTIES TIMEOUT 120)

set_tests_properties(test_bicubic_interp_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_deformable_conv_op PROPERTIES TIMEOUT 200)
set_tests_properties(test_nearest_interp_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_profiler PROPERTIES TIMEOUT 120)
set_tests_properties(test_inplace_softmax_with_cross_entropy PROPERTIES TIMEOUT
                                                                        120)
set_tests_properties(test_cross_entropy2_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_cross_entropy_loss PROPERTIES TIMEOUT 180)
set_tests_properties(test_gru_unit_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_activation_nn_grad PROPERTIES TIMEOUT 250)
set_tests_properties(test_empty_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_parallel_executor_transformer PROPERTIES TIMEOUT 120)
set_tests_properties(test_elementwise_div_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_regularizer_api PROPERTIES TIMEOUT 150)
set_tests_properties(test_multiclass_nms_op PROPERTIES TIMEOUT 120)
if(NOT WIN32)
  if(WITH_NV_JETSON)
    set_tests_properties(test_ir_memory_optimize_nlp PROPERTIES TIMEOUT 1200)
  endif()
endif()
set_tests_properties(test_add_reader_dependency PROPERTIES TIMEOUT 120)
set_tests_properties(test_bilateral_slice_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_buffer_shared_memory_reuse_pass PROPERTIES TIMEOUT
                                                                     120)
set_tests_properties(test_fuse_relu_depthwise_conv_pass PROPERTIES TIMEOUT 120)
set_tests_properties(test_fleet_util PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_transformer_sorted_gradient
                     PROPERTIES TIMEOUT 120)
set_tests_properties(test_matmul_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_nearest_interp_v2_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_trilinear_interp_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_bicubic_interp_v2_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_gather_op PROPERTIES TIMEOUT 180)
set_tests_properties(test_static_save_load PROPERTIES TIMEOUT 250)
set_tests_properties(test_pylayer_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_paddle_save_load_binary PROPERTIES TIMEOUT 120)
if(WIN32)
  set_tests_properties(test_static_save_load_large PROPERTIES TIMEOUT 900)
  set_tests_properties(test_paddle_save_load PROPERTIES TIMEOUT 250)
  set_tests_properties(test_cdist PROPERTIES TIMEOUT 250)
else()
  set_tests_properties(test_static_save_load_large PROPERTIES TIMEOUT 600)
  set_tests_properties(test_paddle_save_load PROPERTIES TIMEOUT 250)
  set_tests_properties(test_cdist PROPERTIES TIMEOUT 150)
endif()
if(WITH_NV_JETSON)
  set_tests_properties(test_concat_op PROPERTIES TIMEOUT 1200)
  set_tests_properties(test_conv3d_transpose_part2_op PROPERTIES TIMEOUT 1200)
  set_tests_properties(test_conv3d_op PROPERTIES TIMEOUT 1200)
  set_tests_properties(test_norm_op PROPERTIES TIMEOUT 1200)
  set_tests_properties(test_batch_norm_op_prim_nchw PROPERTIES TIMEOUT 1500)
  set_tests_properties(test_batch_norm_op_prim_nhwc PROPERTIES TIMEOUT 1500)
  set_tests_properties(test_layer_norm_op PROPERTIES TIMEOUT 1500)
  set_tests_properties(test_pool3d_op PROPERTIES TIMEOUT 1500)
else()
  set_tests_properties(test_concat_op PROPERTIES TIMEOUT 400)
  set_tests_properties(test_conv3d_transpose_part2_op PROPERTIES TIMEOUT 120)
  set_tests_properties(test_conv3d_op PROPERTIES TIMEOUT 120)
  set_tests_properties(test_norm_op PROPERTIES TIMEOUT 150)
  set_tests_properties(test_batch_norm_op_prim_nchw PROPERTIES TIMEOUT 250)
  set_tests_properties(test_batch_norm_op_prim_nhwc PROPERTIES TIMEOUT 250)
  set_tests_properties(test_layer_norm_op PROPERTIES TIMEOUT 250)
  set_tests_properties(test_pool3d_op PROPERTIES TIMEOUT 150)
endif()
if((WITH_GPU OR WITH_ROCM) AND (LINUX))
  py_test_modules(test_conv3d_transpose_op MODULES test_conv3d_transpose_op
                  ENVS NVIDIA_TF32_OVERRIDE=0)
  set_tests_properties(test_conv3d_transpose_op PROPERTIES TIMEOUT 120)
else()
  py_test_modules(test_conv3d_transpose_op MODULES test_conv3d_transpose_op)
  if(WITH_NV_JETSON)
    set_tests_properties(test_conv3d_transpose_op PROPERTIES TIMEOUT 1200)
  else()
    set_tests_properties(test_conv3d_transpose_op PROPERTIES TIMEOUT 120)
  endif()
endif()
set_tests_properties(test_imperative_selected_rows_to_lod_tensor
                     PROPERTIES TIMEOUT 200)
set_tests_properties(test_index_select_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_index_add_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_tensordot PROPERTIES TIMEOUT 200)
set_tests_properties(test_partial_eager_deletion_transformer PROPERTIES TIMEOUT
                                                                        120)
set_tests_properties(test_parallel_executor_seresnext_with_reduce_gpu
                     PROPERTIES TIMEOUT 120)
set_tests_properties(test_dropout_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_argsort_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_gather_nd_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_nn_grad PROPERTIES TIMEOUT 180)
set_tests_properties(test_elementwise_sub_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_row_conv_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_parallel_executor_seresnext_with_fuse_all_reduce_gpu
                     PROPERTIES TIMEOUT 120)
set_tests_properties(test_distributed_fused_lamb_op_with_clip PROPERTIES TIMEOUT
                                                                         240)
set_tests_properties(test_distributed_fused_lamb_op_without_clip
                     PROPERTIES TIMEOUT 120)
set_tests_properties(test_distributed_fused_lamb_op_with_gradient_merge
                     PROPERTIES TIMEOUT 240)
set_tests_properties(test_elementwise_min_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_nan_inf PROPERTIES TIMEOUT 120)
set_tests_properties(test_deformable_conv_v1_op PROPERTIES TIMEOUT 300)
set_tests_properties(test_parallel_executor_transformer_auto_growth
                     PROPERTIES TIMEOUT 120)
set_tests_properties(test_elementwise_add_op PROPERTIES TIMEOUT 200)
if(NOT WITH_COVERAGE)
  set_tests_properties(test_weight_decay PROPERTIES TIMEOUT 120)
endif()
set_tests_properties(test_imperative_ptb_rnn_sorted_gradient PROPERTIES TIMEOUT
                                                                        120)
set_tests_properties(test_paddlescience PROPERTIES TIMEOUT 120)
set_tests_properties(test_crop_tensor_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_ptb_rnn PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_save_load_v2 PROPERTIES TIMEOUT 120)
set_tests_properties(test_conv2d_transpose_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_multiprocess_dataloader_iterable_dataset_static
                     PROPERTIES TIMEOUT 120)
set_tests_properties(test_lstm_cudnn_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_stack_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_bilinear_interp_v2_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_svd_op PROPERTIES TIMEOUT 80)
set_tests_properties(test_einsum_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_qr_op PROPERTIES TIMEOUT 60)
set_tests_properties(test_trilinear_interp_v2_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_masked_select_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_sigmoid_cross_entropy_with_logits_op
                     PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_optimizer_v2 PROPERTIES TIMEOUT 150)
set_tests_properties(test_partial_sum_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_cond PROPERTIES TIMEOUT 120)
set_tests_properties(test_sgd_op PROPERTIES TIMEOUT 250)
set_tests_properties(test_parallel_executor_seresnext_base_gpu
                     PROPERTIES TIMEOUT 120)
set_tests_properties(test_norm_nn_grad PROPERTIES TIMEOUT 180)
set_tests_properties(test_matrix_nms_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_generator_dataloader PROPERTIES TIMEOUT 120)
set_tests_properties(test_partial_concat_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_fuse_optimizer_pass PROPERTIES TIMEOUT 120)
set_tests_properties(test_softmax_with_cross_entropy_op PROPERTIES TIMEOUT 220)
set_tests_properties(test_reduce_op PROPERTIES TIMEOUT 500)
set_tests_properties(test_adam_optimizer_fp32_fp64 PROPERTIES TIMEOUT 120)
set_tests_properties(test_elementwise_nn_grad PROPERTIES TIMEOUT 120)
set_tests_properties(
  test_buffer_shared_memory_reuse_pass_and_fuse_optimization_op_pass
  PROPERTIES TIMEOUT 120)
set_tests_properties(test_conv_nn_grad PROPERTIES TIMEOUT 220)
set_tests_properties(test_program_prune_backward PROPERTIES TIMEOUT 120)
set_tests_properties(test_group_norm_op PROPERTIES TIMEOUT 1000)
set_tests_properties(test_imperative_optimizer PROPERTIES TIMEOUT 250)
set_tests_properties(test_imperative_optimizer_v2 PROPERTIES TIMEOUT 250)
set_tests_properties(test_pool2d_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_transpose_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_activation_op PROPERTIES TIMEOUT 270)
set_tests_properties(test_normal PROPERTIES TIMEOUT 120)
set_tests_properties(test_bilinear_interp_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_decoupled_py_reader PROPERTIES TIMEOUT 120)
set_tests_properties(test_fuse_bn_act_pass PROPERTIES TIMEOUT 120)
set_tests_properties(test_conv2d_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_conv2d_op_depthwise_conv PROPERTIES TIMEOUT 120)
set_tests_properties(test_conv2d_api PROPERTIES TIMEOUT 120)
set_tests_properties(test_elementwise_mul_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_cyclic_cifar_dataset PROPERTIES TIMEOUT 120)
set_tests_properties(test_fuse_all_reduce_pass PROPERTIES TIMEOUT 120)
set_tests_properties(test_dygraph_multi_forward PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_ocr_attention_model PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_mnist PROPERTIES TIMEOUT 120)
set_tests_properties(test_fused_elemwise_activation_op PROPERTIES TIMEOUT 270)
set_tests_properties(test_fused_elemwise_activation_op
                     PROPERTIES LABELS "RUN_TYPE=NIGHTLY")
set_tests_properties(test_gru_op PROPERTIES TIMEOUT 200)
set_tests_properties(test_regularizer PROPERTIES TIMEOUT 150)
set_tests_properties(test_imperative_resnet PROPERTIES TIMEOUT 200)
set_tests_properties(test_imperative_resnet_sorted_gradient PROPERTIES TIMEOUT
                                                                       200)
set_tests_properties(test_imperative_se_resnext PROPERTIES TIMEOUT 200)
set_tests_properties(test_matmul_v2_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_slice_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_strided_slice_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_translated_layer PROPERTIES TIMEOUT 120)
set_tests_properties(test_pad3d_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_dataloader_keep_order PROPERTIES TIMEOUT 120)
set_tests_properties(test_mean_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_dataloader_unkeep_order PROPERTIES TIMEOUT 120)
set_tests_properties(test_reader_reset PROPERTIES TIMEOUT 120)
set_tests_properties(test_pool3d_api PROPERTIES TIMEOUT 120)
set_tests_properties(test_cumprod_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_split_program PROPERTIES TIMEOUT 120)
set_tests_properties(test_graph_send_ue_recv_op PROPERTIES TIMEOUT 60)
set_tests_properties(test_graph_send_uv_op PROPERTIES TIMEOUT 60)

set_tests_properties(test_dataset_cifar PROPERTIES TIMEOUT 120)
set_tests_properties(test_pretrained_model PROPERTIES TIMEOUT 120)
set_tests_properties(test_model PROPERTIES TIMEOUT 300)
set_tests_properties(test_dataset_movielens PROPERTIES TIMEOUT 120)
set_tests_properties(test_datasets PROPERTIES TIMEOUT 300)
set_tests_properties(test_dataset_wmt PROPERTIES TIMEOUT 120)
set_tests_properties(test_vision_models PROPERTIES TIMEOUT 120)
set_tests_properties(test_dataset_uci_housing PROPERTIES TIMEOUT 120)
set_tests_properties(test_dataset_imdb PROPERTIES TIMEOUT 300)
set_tests_properties(test_pretrained_model PROPERTIES TIMEOUT 600)
set_tests_properties(test_callback_wandb PROPERTIES TIMEOUT 60)
if(WITH_COVERAGE)
  set_tests_properties(test_hapi_hub PROPERTIES TIMEOUT 300)
endif()

if(APPLE)
  set_tests_properties(test_callback_early_stop PROPERTIES TIMEOUT 300)
  set_tests_properties(test_callback_reduce_lr_on_plateau PROPERTIES TIMEOUT
                                                                     300)
  set_tests_properties(test_vision_models PROPERTIES TIMEOUT 300)
endif()
set_tests_properties(test_decorator PROPERTIES TIMEOUT 120)

if(WITH_DISTRIBUTE
   AND WITH_GPU
   AND WITH_NCCL)
  set_tests_properties(test_auto_parallel_data_unshard PROPERTIES TIMEOUT 120)
  set_tests_properties(test_auto_parallel_save_load PROPERTIES TIMEOUT 120)
  set_tests_properties(test_auto_parallel_autoconvert PROPERTIES TIMEOUT 120)
endif()

if(APPLE)
  set_tests_properties(test_imperative_transformer_sorted_gradient
                       PROPERTIES TIMEOUT 300)
  set_tests_properties(test_multiclass_nms_op PROPERTIES TIMEOUT 300)
  set_tests_properties(test_weight_decay PROPERTIES TIMEOUT 300)
endif()

if((WITH_ROCM OR WITH_GPU) AND NOT WIN32)
  if(WITH_DISTRIBUTE)
    set_tests_properties(test_static_model_parallel_fused_feedforward
                         PROPERTIES TIMEOUT 120)
    set_tests_properties(test_static_model_parallel_fused_attention
                         PROPERTIES TIMEOUT 120)
    set_tests_properties(test_static_model_parallel_fused_multi_transformer
                         PROPERTIES TIMEOUT 120)
    set_tests_properties(test_pipeline_parallel PROPERTIES LABELS
                                                           "RUN_TYPE=DIST")
    set_tests_properties(test_fleet_perf_test PROPERTIES LABELS "RUN_TYPE=DIST")
    set_tests_properties(test_reducescatter PROPERTIES TIMEOUT 120)
    set_tests_properties(test_allgather PROPERTIES TIMEOUT 120)
  endif()
  set_tests_properties(test_paddle_multiprocessing PROPERTIES TIMEOUT 120)
  set_tests_properties(test_pipeline_parallel PROPERTIES TIMEOUT 120)
  set_tests_properties(test_fleet_perf_test PROPERTIES TIMEOUT 120)
endif()
if(WITH_GPU OR WITH_ROCM)
  set_tests_properties(test_rank_attention_op PROPERTIES TIMEOUT 120)
endif()
if(WITH_GPU AND NOT WIN32)
  set_tests_properties(test_fused_multi_transformer_int8_op PROPERTIES TIMEOUT
                                                                       60)
endif()
set_tests_properties(test_inplace_addto_strategy PROPERTIES TIMEOUT 120)
set_tests_properties(test_eigvals_op PROPERTIES TIMEOUT 400)
set_tests_properties(
  test_cuda_memory_reserved PROPERTIES ENVIRONMENT
                                       "FLAGS_allocator_strategy=auto_growth")
if(WITH_GLOO)
  set_tests_properties(test_parallel_dygraph_dataparallel_cpuonly
                       PROPERTIES TIMEOUT 30)
  set_tests_properties(test_parallel_dygraph_unused_variables_gloo
                       PROPERTIES TIMEOUT 120)
  set_tests_properties(test_parallel_dygraph_sparse_embedding_gloo
                       PROPERTIES TIMEOUT 120)
  set_tests_properties(test_parallel_dygraph_sparse_embedding_over_height_gloo
                       PROPERTIES TIMEOUT 120)
endif()

set(TEST_CINN_OPS
    test_softmax_op
    test_expand_v2_op
    test_reduce_op
    test_slice_op
    test_stack_op
    test_activation_op
    test_full_like_op
    test_index_select_op
    test_fill_any_like_op
    test_concat_op
    test_top_k_v2_op
    test_elementwise_add_op
    test_elementwise_sub_op
    test_elementwise_div_op
    test_elementwise_mul_op
    test_gather_nd_op
    test_squeeze2_op
    test_elementwise_pow_op
    test_elementwise_max_op
    test_transpose_op
    test_reshape_op
    test_mean_op
    test_unsqueeze2_op
    test_meshgrid_op
    test_scale_op
    test_clip_op
    test_scatter_op
    test_gather_op
    test_batch_norm_op_prim_nchw
    test_batch_norm_op_prim_nhwc
    test_layer_norm_op
    test_cast_op
    test_dropout_op
    test_group_norm_op
    test_tile_op
    test_roll_op
    test_sum_op
    test_elementwise_min_op
    test_atan2_op
    test_top_k_op
    test_where_op
    test_take_along_axis_op
    test_arg_min_max_op
    test_reverse_op
    test_flip
    test_triangular_solve_op
    test_scatter_nd_op
    test_strided_slice_op
    test_pool2d_op
    test_instance_norm_op
    test_cumsum_op
    test_pad_op
    test_split_op
    test_erf_op
    test_assign_op
    test_flatten_contiguous_range_op)

foreach(TEST_CINN_OP ${TEST_CINN_OPS})
  if(WITH_CINN)
    set_tests_properties(${TEST_CINN_OP} PROPERTIES LABELS "RUN_TYPE=CINN")

    get_test_property(${TEST_CINN_OP} TIMEOUT ORIGIN_TIME_OUT)
    if((NOT ${ORIGIN_TIME_OUT}) OR (${ORIGIN_TIME_OUT} LESS 200))
      set_tests_properties(${TEST_CINN_OP} PROPERTIES TIMEOUT 200)
    endif()
  endif()
endforeach()

if(WITH_CINN AND WITH_TESTING)
  set_tests_properties(
    test_parallel_executor_run_cinn
    PROPERTIES
      LABELS
      "RUN_TYPE=CINN"
      ENVIRONMENT
      FLAGS_allow_cinn_ops="conv2d;conv2d_grad;elementwise_add;elementwise_add_grad;relu;relu_grad;sum"
  )
endif()

set_tests_properties(
  test_cuda_graph_static_mode
  PROPERTIES ENVIRONMENT "FLAGS_CUDA_GRAPH_USE_STANDALONE_EXECUTOR=1")
set_tests_properties(
  test_cuda_graph_static_mode_error
  PROPERTIES ENVIRONMENT "FLAGS_CUDA_GRAPH_USE_STANDALONE_EXECUTOR=1")
# In test_conditional_block, the sub block changes the dtype and place of the output variable.
# The changed variable is used in the following op. Static build is not supported for this case.
set_tests_properties(test_conditional_block
                     PROPERTIES ENVIRONMENT "FLAGS_new_executor_static_build=0")

# These UTs are to temporarily test static build for standalone_executor, will be removed after static build is enabled by default.
set(STATIC_BUILD_TESTS
    test_adagrad_op
    test_adamw_op
    test_arg_min_max_op
    test_assign_pos_op
    test_batch_norm_op
    test_bucketize_api
    test_bincount_op
    test_c_embedding_op
    test_decayed_adagrad_op
    test_decoupled_py_reader
    test_eig_op
    test_eigh_op
    test_fake_dequantize_op
    test_fake_quantize_op
    test_fetch_lod_tensor_array
    test_ftrl_op
    test_fused_attention_op
    test_fused_attention_op_api
    test_fuse_bn_act_pass
    test_fused_feedforward_op
    test_fused_feedforward_pass
    test_fused_layernorm_op
    test_imperative_optimizer
    test_lamb_op
    test_layer_norm_op
    test_limit_by_capacity_op
    test_lookup_table_bf16_op
    test_lookup_table_v2_op
    test_matmul_op
    test_matmul_v2_op
    test_merged_adam_op
    test_momentum_op
    test_nce
    test_number_count_op
    test_paddle_save_load_binary
    test_prune_gate_by_capacity_op
    test_random_routing_op
    test_reduce_op
    test_searchsorted_op
    test_segment_ops
    test_sparse_momentum_op
    test_sgd_op_bf16
    test_shuffle_batch_op
    test_softmax_mask_fuse_upper_triangle_op
    test_sparse_conv_op
    test_sparse_norm_op
    test_sparse_pooling_op
    test_sync_batch_norm_op
    test_tensor_array_to_tensor
    test_unique
    test_update_loss_scaling_op
    test_while_op
    test_one_hot_v2_op)

if(NOT WITH_GPU)
  list(REMOVE_ITEM STATIC_BUILD_TESTS test_fused_attention_op)
  list(REMOVE_ITEM STATIC_BUILD_TESTS test_fused_attention_op_api)
  list(REMOVE_ITEM STATIC_BUILD_TESTS test_fused_feedforward_op)
  list(REMOVE_ITEM STATIC_BUILD_TESTS test_fused_feedforward_op_pass)
endif()

if(((NOT WITH_ROCM) AND (NOT WITH_GPU)) OR WIN32)
  list(REMOVE_ITEM STATIC_BUILD_TESTS test_c_embedding_op)
endif()

foreach(STATIC_BUILD_TEST ${STATIC_BUILD_TESTS})
  py_test_modules(
    ${STATIC_BUILD_TEST}_static_build MODULES ${STATIC_BUILD_TEST} ENVS
    FLAGS_new_executor_static_build=true)
endforeach()

set(PIR_COVERAGE_TESTS test_fused_feedforward_pass
                       test_fuse_elewise_add_act_pass)
if(NOT WITH_GPU)
  list(REMOVE_ITEM PIR_COVERAGE_TESTS test_fused_feedforward_pass)
endif()

if(APPLE)
  list(REMOVE_ITEM PIR_COVERAGE_TESTS test_fuse_elewise_add_act_pass)
endif()

foreach(PIR_COVERAGE_TEST ${PIR_COVERAGE_TESTS})
  py_test_modules(${PIR_COVERAGE_TEST}_pir MODULES ${PIR_COVERAGE_TEST} ENVS
                  FLAGS_enable_pir_in_executor=true)
  set_tests_properties(${PIR_COVERAGE_TEST}_pir PROPERTIES TIMEOUT 120)
  message(STATUS "PIR Copied OpTest: ${PIR_COVERAGE_TEST}_pir in legacy_test")
endforeach()

set_tests_properties(test_decoupled_py_reader_static_build PROPERTIES TIMEOUT
                                                                      120)
set_tests_properties(test_fuse_bn_act_pass_static_build PROPERTIES TIMEOUT 120)
set_tests_properties(
  test_fuse_bn_act_pass_static_build
  PROPERTIES
    ENVIRONMENT
    "FLAGS_cudnn_deterministic=1;FLAGS_cudnn_batchnorm_spatial_persistent=1;FLAGS_conv_workspace_size_limit=1000"
)
set_tests_properties(test_imperative_optimizer_static_build PROPERTIES TIMEOUT
                                                                       250)
set_tests_properties(test_matmul_op_static_build PROPERTIES TIMEOUT 120)
set_tests_properties(test_matmul_v2_op_static_build PROPERTIES TIMEOUT 120)
set_tests_properties(test_layer_norm_op_static_build PROPERTIES TIMEOUT 1500)
set_tests_properties(test_paddle_save_load_binary_static_build
                     PROPERTIES TIMEOUT 120)
set_tests_properties(test_reduce_op_static_build PROPERTIES TIMEOUT 500)
set_tests_properties(test_sync_batch_norm_op_static_build
                     PROPERTIES LABELS "RUN_TYPE=DIST")
set_tests_properties(test_sync_batch_norm_op_static_build PROPERTIES TIMEOUT
                                                                     250)
py_test_modules(test_stride MODULES test_stride ENVS
                FLAGS_use_stride_kernel=true)

if((WITH_ROCM OR WITH_GPU) AND NOT WIN32)
  # These UTs are specially designed for FleetExecutor
  set_tests_properties(
    test_fleet_executor PROPERTIES ENVIRONMENT
                                   "FLAGS_new_executor_micro_batching=False")
  set_tests_properties(
    test_fleet_executor_origin_scheduler
    PROPERTIES ENVIRONMENT "FLAGS_new_executor_micro_batching=False")
  set_tests_properties(
    test_fleet_executor_with_task_nodes
    PROPERTIES ENVIRONMENT "FLAGS_new_executor_micro_batching=False")
endif()

set_pit_tests_properties()
